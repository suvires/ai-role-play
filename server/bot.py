import os

# Use cached Hugging Face models only (faster startup). Must run before any import that
# loads huggingface_hub. Run with HF_HUB_OFFLINE=0 the first time, or to download new models.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Dict, Optional

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v2 import LocalSmartTurnAnalyzerV2
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService

from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

from clause_text_aggregator import FirstClauseTextAggregator
from stt_parakeet_mlx import ParakeetSTTServiceMLX
from tts_mlx_isolated import TTSMLXIsolated
from camera_vision import CameraVisionProcessor
from emotion_tags import EmotionTagProcessor
from memory import load_notes, update_notes
from personas import DEFAULT_PERSONA, PERSONAS, Persona, get_persona

load_dotenv(override=True)

app = FastAPI()

pcs_map: Dict[str, SmallWebRTCConnection] = {}
_background_tasks: set = set()  # memory updates that outlive their conversation

TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
# Voice used to warm up spare TTS workers; each persona sets its own (personas.py).
# Spanish voices: "ef_dora" (female), "em_alex" / "em_santa" (male)
TTS_VOICE = "ef_dora"

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]


SYSTEM_INSTRUCTION = """
{personality} {language}

Tienes voz: todo lo que escribes se lee en voz alta al usuario, que te está escuchando. Nunca digas que no puedes hablar o que no tienes voz. Si te piden recitar, leer o contar algo, hazlo directamente. Si te piden cantar, explica con naturalidad que tu voz solo sabe hablar y ofrécete a recitarlo.

{camera}

Tienes una cara que el usuario ve. Empieza siempre cada respuesta con una etiqueta entre corchetes que indique la emoción con la que hablas: [neutral], [alegre], [relajado], [triste], [enfadado] o [sorprendido]. Elige la que encaje con lo que dices y con lo que te cuentan. La etiqueta no se lee en voz alta, solo cambia la expresión de tu cara. Escríbela solo al principio de la respuesta y siempre tal cual, en español, aunque hables en otro idioma.

Tu entrada es texto transcrito en tiempo real a partir de la voz del usuario. Puede haber errores de transcripción. Ajusta tus respuestas automáticamente para tenerlos en cuenta.

Tu respuesta se convertirá en audio, así que no incluyas caracteres especiales, emojis, markdown ni ningún tipo de formato. Escribe los números y las abreviaturas con letras.

Habla como una persona cercana y con curiosidad en una conversación de verdad, no como un buscador. Ajusta la longitud a lo que pide el usuario: a un saludo o una pregunta sencilla responde con naturalidad en dos o tres frases, y cuando el tema da para más, desarróllalo con ejemplos, detalles o alguna anécdota. Si encaja, termina con una pregunta o un comentario que invite a seguir hablando. Como te van a escuchar y no a leer, usa frases de longitud natural, fáciles de seguir de oído.

Si te piden algo largo, como un poema, un cuento o una explicación detallada, hazlo completo, sin resumirlo ni remitir a otra fuente. Cuando recites poesía, pon cada verso en una línea distinta. Si no recuerdas un texto exacto, dilo con sinceridad y ofrece uno escrito por ti, sin atribuirlo a un autor real.

{greeting}
"""


def build_prompt(persona: Persona, camera: str) -> str:
    """The persona's prompt, with what it remembers from previous conversations."""
    notes = load_notes(persona.id)
    if notes:
        greeting = (
            f"Ya has hablado otras veces con este usuario. Esto es lo que recuerdas de él:\n{notes}\n\n"
            "Empieza la conversación saludándole como alguien que ya le conoce, mencionando con "
            "naturalidad algo de lo que recuerdas, en una o dos frases, y luego espera a que hable."
        )
    else:
        greeting = f'Empieza la conversación diciendo "{persona.greeting}" y luego espera a que hable el usuario.'
    return SYSTEM_INSTRUCTION.format(
        personality=persona.personality, language=persona.language, camera=camera, greeting=greeting
    )

# Small local models can't reliably use tools: the camera frame is attached when the
# user's words sound visual (CameraVisionProcessor). Claude decides itself, with a tool.
CAMERA_WITH_KEYWORDS = """También puedes ver al usuario a través de su cámara. Cuando su mensaje venga acompañado de una imagen, es lo que su cámara ve en ese momento: úsala para responder con naturalidad, como si lo tuvieras delante, sin decir que es una imagen o una foto. Si te pregunta algo visual y no te llega ninguna imagen, dile que ahora mismo no ves su cámara."""

CAMERA_WITH_TOOL = """También puedes ver al usuario a través de su cámara con la herramienta mirar_camara. Úsala siempre que tu respuesta dependa de algo que el usuario te enseña o de cómo está él o lo que le rodea (lo que tiene en la mano, su ropa, su cara, lo que hay detrás), aunque no te lo pida con esas palabras, por ejemplo "¿qué te parece esto?". Antes de usarla di algo muy breve, como "A ver...", para que no haya un silencio. Mira de nuevo cada vez que te pregunte algo visual, porque lo que ve la cámara cambia. Responde a lo que ves con naturalidad, como si lo tuvieras delante, sin decir que es una imagen o una foto. Si la herramienta dice que la cámara está apagada, díselo."""

LOOK_AT_CAMERA = FunctionSchema(
    name="mirar_camara",
    description="Mira por la cámara del usuario y te devuelve lo que se ve ahora mismo.",
    properties={
        "pregunta": {"type": "string", "description": "Qué quieres averiguar al mirar, en español."}
    },
    required=["pregunta"],
)


def create_llm():
    """LLM_PROVIDER in .env: "local" (LM Studio, default) or "anthropic" (Claude, needs ANTHROPIC_API_KEY)."""
    provider = os.getenv("LLM_PROVIDER", "local")
    if provider == "anthropic":
        return AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
            params=AnthropicLLMService.InputParams(
                extra={
                    # Haiku 5.5 thinks by default; turn it off so replies start right away
                    "thinking": {"type": "disabled"},
                    # Automatic prompt caching: the system prompt, tools and the growing
                    # conversation are re-read from cache on every turn
                    "extra_body": {"cache_control": {"type": "ephemeral"}},
                }
            ),
        )
    if provider != "local":
        raise ValueError(f'LLM_PROVIDER must be "local" or "anthropic", not "{provider}"')

    return OpenAILLMService(
        api_key="dummyKey",
        model="google/gemma-4-e4b",  # Small model. Good Spanish. Uses ~7GB of RAM.
        # model="google/gemma-4-26b-a4b",  # MoE (~4B active). Better Spanish. Uses ~15GB of RAM.
        # model="gemma-3n-e4b-it-text",  # Small model. Uses ~4GB of RAM.
        # model="google/gemma-3-12b",  # Medium-sized model. Uses ~8.5GB of RAM.
        # model="mlx-community/Qwen3-235B-A22B-Instruct-2507-3bit-DWQ", # Large model. Uses ~110GB of RAM!
        base_url="http://127.0.0.1:1234/v1",
        max_tokens=4096,
        # Gemma 4 "thinks" before answering, adding ~20s of latency. Disable it.
        params=OpenAILLMService.InputParams(extra={"reasoning_effort": "none"}),
    )


def create_tts(persona: Persona):
    """TTS_PROVIDER in .env: "kokoro" (local, default) or "cartesia" (needs CARTESIA_API_KEY and
    CARTESIA_VOICE_ID). Both start speaking at the first comma of a long first sentence."""
    provider = os.getenv("TTS_PROVIDER", "kokoro")
    if provider == "cartesia":
        return CartesiaTTSService(
            api_key=os.environ["CARTESIA_API_KEY"],
            voice_id=persona.cartesia_voice or os.environ["CARTESIA_VOICE_ID"],
            model=os.getenv("CARTESIA_MODEL", "sonic-3"),
            # The persona's Kokoro voice tells its language ("a..." voices are US English)
            params=CartesiaTTSService.InputParams(
                language=Language.EN if persona.voice.startswith("a") else Language.ES
            ),
            text_aggregator=FirstClauseTextAggregator(),
        )
    if provider != "kokoro":
        raise ValueError(f'TTS_PROVIDER must be "kokoro" or "cartesia", not "{provider}"')

    return TTSMLXIsolated(
        model=TTS_MODEL,
        voice=persona.voice,
        sample_rate=24000,
        text_aggregator=FirstClauseTextAggregator(),
    )
    # return TTSMLXIsolated(model="Marvis-AI/marvis-tts-250m-v0.1", voice=None)


async def run_bot(webrtc_connection, persona: Persona):
    logger.info(f"Starting a conversation with {persona.name} ({persona.id})")
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=True,  # user's camera, see CameraVisionProcessor
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV2(
                smart_turn_model_path="",  # Download from HuggingFace
                params=SmartTurnParams(),
            ),
        ),
    )

    stt = ParakeetSTTServiceMLX(language=Language.ES)  # ~0.15s per turn
    # stt = WhisperSTTServiceMLX(model=MLXModel.LARGE_V3_TURBO_Q4, language=Language.ES)  # ~1s per turn

    tts = create_tts(persona)

    llm = create_llm()

    camera = CameraVisionProcessor(auto_attach=not isinstance(llm, AnthropicLLMService))
    if isinstance(llm, AnthropicLLMService):
        # A real system prompt (cached), plus a first turn that triggers the greeting
        context = OpenAILLMContext(
            [
                {"role": "system", "content": build_prompt(persona, CAMERA_WITH_TOOL)},
                {"role": "user", "content": "(El usuario se acaba de conectar.)"},
            ],
            tools=ToolsSchema(standard_tools=[LOOK_AT_CAMERA]),
        )

        async def look_at_camera(params: FunctionCallParams):
            if not camera.has_recent_frame():
                await params.result_callback({"error": "La cámara del usuario está apagada."})
                return
            # The transport answers with the next camera frame; the context aggregator adds
            # it to the conversation and runs the LLM again
            await params.llm.request_image_frame(
                user_id=webrtc_connection.pc_id,
                function_name=params.function_name,
                tool_call_id=params.tool_call_id,
                text_content=params.arguments.get("pregunta"),
            )

        llm.register_function("mirar_camara", look_at_camera)
    else:
        # LM Studio's Gemma template has no system role: the prompt goes as the first user turn
        context = OpenAILLMContext(
            [{"role": "user", "content": build_prompt(persona, CAMERA_WITH_KEYWORDS)}]
        )
    context_aggregator = llm.create_context_aggregator(
        context,
        # Whisper local service isn't streaming, so it delivers the full text all at
        # once, after the UserStoppedSpeaking frame. Set aggregation_timeout to a
        # a de minimus value since we don't expect any transcript aggregation to be
        # necessary.
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.05),
    )

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rtvi,
            context_aggregator.user(),
            camera,
            llm,
            EmotionTagProcessor(),
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Conversation with {persona.name} ended")
        # Remember the conversation in the background, then stop the pipeline
        background = asyncio.create_task(
            update_notes(persona.id, persona.name, list(context_aggregator.user().context.messages))
        )
        _background_tasks.add(background)
        background.add_done_callback(_background_tasks.discard)
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


@app.get("/api/personas")
async def personas():
    return {"default": DEFAULT_PERSONA, "personas": [p.public() for p in PERSONAS.values()]}


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks, persona: Optional[str] = None):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        background_tasks.add_task(run_bot, pipecat_connection, get_persona(persona))

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()

    # Load the local TTS model now so the first connection doesn't wait for it
    if os.getenv("TTS_PROVIDER", "kokoro") == "kokoro":
        TTSMLXIsolated.prewarm_in_background(TTS_MODEL, TTS_VOICE)

    uvicorn.run(app, host=args.host, port=args.port)
