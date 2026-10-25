"use client";

import { useEffect, useMemo, useState } from "react";
import type { PipecatClientOptions } from "@pipecat-ai/client-js";
import { usePipecatClientTransportState } from "@pipecat-ai/client-react";
import {
  ConnectButton,
  PipecatAppBase,
  TranscriptOverlay,
  UserAudioControl,
  UserVideoControl,
} from "@pipecat-ai/voice-ui-kit";

import { useConversationLog } from "./useConversationLog";
import VrmAvatar from "./VrmAvatar";

// Styles: src/app/avatar-app.css (global, imported by layout.tsx so it is in the page from the start)
const styles = Object.fromEntries(
  [
    "main", "avatar", "avatarPicker", "avatarOption", "avatarSelected", "emotion", "persona",
    "transcript", "controls", "camera", "download", "error",
  ].map((name) => [name, `avatar-app-${name}`]),
);

// Module-level constant: PipecatAppBase recreates the client when its props change
const CLIENT_OPTIONS = { enableMic: true, enableCam: true } as PipecatClientOptions;

// Avatars and their licenses: public/avatars/LICENSES.md
// (bump ?v= after re-converting a model so the browser doesn't keep the old one)
const AVATARS = [
  { url: "/avatars/rocketbox-f08.vrm?v=3", thumb: "/avatars/thumbs/rocketbox-f08.png", name: "Rocketbox (realista)" },
  { url: "/avatars/avatar-a.vrm", thumb: "/avatars/thumbs/avatar-a.png", name: "VRoid AvatarSample A" },
  { url: "/avatars/sample.vrm", thumb: "/avatars/thumbs/sample.png", name: "pixiv VRM sample" },
];

const EMOTION_NAMES: Record<string, string> = {
  neutral: "neutral",
  happy: "alegre",
  relaxed: "relajado",
  sad: "triste",
  angry: "enfadado",
  surprised: "sorprendido",
};

// Characters come from the bot (server/personas.py)
type Persona = { id: string; name: string; description: string; avatar: string };

function remembered(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null; // storage unavailable (private mode...)
  }
}

function remember(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // not remembered next time, that's all
  }
}

function savedAvatar(): string {
  const url = remembered("avatar-url");
  return url && AVATARS.some((avatar) => avatar.url === url) ? url : AVATARS[0].url;
}

export default function AvatarApp() {
  const [avatarUrl, setAvatarUrl] = useState(savedAvatar);
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [personaId, setPersonaId] = useState(() => remembered("persona") ?? "pipecat");

  useEffect(() => {
    fetch("/api/personas")
      .then((response) => response.json())
      .then(({ personas, default: fallback }: { personas: Persona[]; default: string }) => {
        setPersonas(personas);
        setPersonaId((id) => (personas.some((persona) => persona.id === id) ? id : fallback));
      })
      .catch(() => console.warn("Could not load the characters: is the bot running?"));
  }, []);

  const chooseAvatar = (url: string) => {
    setAvatarUrl(url);
    remember("avatar-url", url);
  };

  const choosePersona = (id: string) => {
    setPersonaId(id);
    remember("persona", id);
    // Switch to the character's avatar (the picker can still change it)
    const avatar = AVATARS.find((a) => a.url.split("?")[0] === personas.find((p) => p.id === id)?.avatar);
    if (avatar) chooseAvatar(avatar.url);
  };

  // The character is chosen when connecting
  const connectParams = useMemo(
    () => ({ connectionUrl: `/api/offer?persona=${encodeURIComponent(personaId)}` }),
    [personaId],
  );

  return (
    <PipecatAppBase
      transportType="smallwebrtc"
      connectParams={connectParams}
      clientOptions={CLIENT_OPTIONS}
      themeProps={{ defaultTheme: "dark" }}
    >
      {({ handleConnect, handleDisconnect, error }) => (
        <Stage
          avatarUrl={avatarUrl}
          onAvatarChange={chooseAvatar}
          personas={personas}
          personaId={personaId}
          onPersonaChange={choosePersona}
          onConnect={handleConnect}
          onDisconnect={handleDisconnect}
          error={error}
        />
      )}
    </PipecatAppBase>
  );
}

type StageProps = {
  avatarUrl: string;
  onAvatarChange: (url: string) => void;
  personas: Persona[];
  personaId: string;
  onPersonaChange: (id: string) => void;
  onConnect?: () => void | Promise<void>;
  onDisconnect?: () => void | Promise<void>;
  error?: string | null;
};

/** The page inside the Pipecat client provider (so it can use the client hooks). */
function Stage({
  avatarUrl,
  onAvatarChange,
  personas,
  personaId,
  onPersonaChange,
  onConnect,
  onDisconnect,
  error,
}: StageProps) {
  const [emotion, setEmotion] = useState("neutral");
  const [pickedEmotion, setPickedEmotion] = useState<{ name: string; id: number } | null>(null);
  const transportState = usePipecatClientTransportState();
  const inConversation = !["disconnected", "initialized", "error"].includes(transportState);
  const log = useConversationLog();
  const persona = personas.find((p) => p.id === personaId);

  // A new conversation starts a new log
  const { clear } = log;
  useEffect(() => {
    if (transportState === "connecting") clear();
  }, [transportState, clear]);

  return (
    <main className={styles.main}>
      <div className={styles.avatar}>
        <VrmAvatar url={avatarUrl} onEmotionChange={setEmotion} pickedEmotion={pickedEmotion} />
      </div>

      {personas.length > 0 && (
        <label
          className={styles.persona}
          title={inConversation ? "Desconecta para cambiar de personaje" : persona?.description}
        >
          Personaje:
          <select
            value={personaId}
            disabled={inConversation}
            onChange={(event) => onPersonaChange(event.target.value)}
          >
            {personas.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} — {p.description}
              </option>
            ))}
          </select>
        </label>
      )}

      <div className={styles.avatarPicker} role="radiogroup" aria-label="Avatar">
        {AVATARS.map((avatar) => (
          <button
            key={avatar.url}
            type="button"
            role="radio"
            aria-checked={avatar.url === avatarUrl}
            className={avatar.url === avatarUrl ? styles.avatarSelected : styles.avatarOption}
            onClick={() => onAvatarChange(avatar.url)}
            title={avatar.name}
          >
            {/* eslint-disable-next-line @next/next/no-img-element -- small static thumbnails */}
            <img src={avatar.thumb} alt={avatar.name} />
          </button>
        ))}
      </div>

      <label className={styles.emotion} title="También con las teclas 1-6">
        Emoción:
        <select value={emotion} onChange={(event) => setPickedEmotion({ name: event.target.value, id: Date.now() })}>
          {Object.entries(EMOTION_NAMES).map(([value, name]) => (
            <option key={value} value={value}>
              {name}
            </option>
          ))}
        </select>
      </label>

      <div className={styles.transcript}>
        <TranscriptOverlay participant="remote" />
      </div>
      {error && <p className={styles.error}>{error}</p>}
      <div className={styles.controls}>
        <UserAudioControl />
        <UserVideoControl classNames={{ container: styles.camera }} />
        <ConnectButton onConnect={onConnect} onDisconnect={onDisconnect} />
        <button
          type="button"
          className={styles.download}
          disabled={log.entries.length === 0}
          onClick={() => log.download(persona?.name ?? "Pipecat")}
          title="Descargar la conversación en un archivo de texto"
        >
          Descargar conversación
        </button>
      </div>
    </main>
  );
}
