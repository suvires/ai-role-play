// Client-only: rendered inside AvatarApp, which page.tsx loads with ssr: false
import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { VRM, VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { RTVIEvent } from "@pipecat-ai/client-js";
import { usePipecatClientMediaTrack, useRTVIClientEvent } from "@pipecat-ai/client-react";

import { LipSync } from "./lipsync";
import { useFaceTracking } from "./useFaceTracking";

type Props = {
  /** URL of the .vrm model, e.g. "/avatars/sample.vrm" */
  url: string;
  /** Called with the VRM expression shown ("neutral", "happy"...) whenever it changes */
  onEmotionChange?: (emotion: string) => void;
  /** Show this expression until another one is picked or the bot sends one (preview) */
  pickedEmotion?: { name: string; id: number } | null;
};

// Emotions sent by the bot (server/emotion_tags.py) and how strongly to show them,
// so the mouth can still move on top of them
const EMOTION_INTENSITY: Record<string, number> = {
  happy: 0.6,
  relaxed: 0.5,
  sad: 0.6,
  angry: 0.55,
  surprised: 0.7,
};
const EMOTION_HOLD_SECS = 4; // keep the face this long after the bot stops talking
// Keys 1-6 show each emotion, to preview them without a conversation
const EMOTION_KEYS = ["neutral", "happy", "relaxed", "sad", "angry", "surprised"];

/** A VRM avatar that blinks, breathes, gestures and moves its lips with the bot's voice. */
export default function VrmAvatar({ url, onEmotionChange, pickedEmotion }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const lipSyncRef = useRef<LipSync | null>(null);
  // The emotion arrives before the bot starts talking, so remember when it arrived.
  // Picked ones (preview) stay until replaced instead of fading back to neutral.
  const emotionRef = useRef({ name: "neutral", at: 0, picked: false });
  const onEmotionChangeRef = useRef(onEmotionChange);
  onEmotionChangeRef.current = onEmotionChange;
  const setEmotion = useCallback((name: string, picked = false) => {
    if (name !== emotionRef.current.name) onEmotionChangeRef.current?.(name);
    emotionRef.current = { name, at: performance.now() / 1000, picked };
  }, []);

  useEffect(() => {
    if (pickedEmotion) setEmotion(pickedEmotion.name, true);
  }, [pickedEmotion, setEmotion]);
  const botAudio = usePipecatClientMediaTrack("audio", "bot");
  // Look the user in the eye (their face in the camera) and react while they talk
  const faceRef = useFaceTracking(usePipecatClientMediaTrack("video", "local"));
  const userTalkingRef = useRef(false);
  useRTVIClientEvent(RTVIEvent.UserStartedSpeaking, useCallback(() => (userTalkingRef.current = true), []));
  useRTVIClientEvent(RTVIEvent.UserStoppedSpeaking, useCallback(() => (userTalkingRef.current = false), []));

  useEffect(() => {
    if (!botAudio) return;
    const lipSync = new LipSync(botAudio);
    lipSyncRef.current = lipSync;
    return () => {
      lipSyncRef.current = null;
      lipSync.dispose();
    };
  }, [botAudio]);

  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    useCallback(
      (data: { type?: string; emotion?: string }) => {
        if (data?.type === "emotion" && data.emotion) setEmotion(data.emotion);
      },
      [setEmotion],
    ),
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const emotion = EMOTION_KEYS[Number(event.key) - 1];
      if (emotion && !(event.target instanceof HTMLInputElement)) setEmotion(emotion, true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setEmotion]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(25, 1, 0.05, 20);
    // Soft portrait lighting: sky/ground fill, a key light from the front-top and a weaker
    // fill from below so the nose and chin don't cast hard dark shadows
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
    keyLight.position.set(0.5, 1.5, 2);
    const fillLight = new THREE.DirectionalLight(0xfff4ea, 0.9);
    fillLight.position.set(-0.4, -0.3, 2);
    scene.add(keyLight, fillLight, new THREE.HemisphereLight(0xffffff, 0x8a7f74, 1.3));

    // Wheel/pinch to zoom (towards the pointer), drag to orbit, right-drag to pan.
    // Double-click goes back to the automatic chest-up framing.
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.zoomToCursor = true;
    controls.minDistance = 0.3;
    controls.maxDistance = 8;
    let userMovedCamera = false;
    controls.addEventListener("start", () => (userMovedCamera = true));
    const resetCamera = () => {
      userMovedCamera = false;
      frameBody();
    };
    renderer.domElement.addEventListener("dblclick", resetCamera);

    // Chest-up framing: from the top of the head (with some headroom) down to the chest,
    // which sits just above the subtitles and the control bar
    let framing: { top: number; chest: number; center: THREE.Vector3 } | null = null;
    const frameBody = () => {
      if (!framing) return;
      const { top, chest, center } = framing;
      const span = (top - chest) / 0.75; // head-to-chest takes the upper 75% of the view
      const viewTop = top + span * 0.12; // headroom
      const target = new THREE.Vector3(center.x, viewTop - span / 2, center.z);
      const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
      const fitHeight = span / 2 / tanHalfFov;
      const fitWidth = 0.6 / 2 / (tanHalfFov * camera.aspect); // keep the shoulders in narrow windows
      camera.position.set(target.x, target.y, center.z + Math.max(fitHeight, fitWidth));
      controls.target.copy(target);
      controls.update();
    };

    const resize = () => {
      const { clientWidth: width, clientHeight: height } = container;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      if (!userMovedCamera) frameBody();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    resize();

    let vrm: VRM | null = null;
    let armPose: ArmPose | null = null;
    let disposed = false;
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    loader.load(url, (gltf) => {
      if (disposed) return;
      vrm = gltf.userData.vrm as VRM;
      VRMUtils.removeUnnecessaryVertices(gltf.scene);
      VRMUtils.combineSkeletons(gltf.scene);
      VRMUtils.rotateVRM0(vrm); // VRM 0.x models face the other way
      armPose = relaxArms(vrm);
      // Let the lips keep moving while the face shows an emotion
      for (const name of Object.keys(EMOTION_INTENSITY)) {
        const expression = vrm.expressionManager?.getExpression(name);
        if (expression) expression.overrideMouth = "none";
      }
      scene.add(vrm.scene);

      // Pose first, then frame from the head top to the chest bone and look at the camera
      vrm.update(0);
      vrm.scene.updateMatrixWorld(true);
      // Cloth/hair physics was set up for the T-pose. With the arms down, gravity slid the
      // sleeves down the arm: make sleeve springs stiff and weightless (hair keeps its
      // physics), and restart the simulation from the relaxed pose so nothing swings.
      for (const joint of vrm.springBoneManager?.joints ?? []) {
        if (/sleeve|upperarm/i.test(joint.bone.name)) {
          joint.settings.gravityPower = 0;
          joint.settings.stiffness = Math.max(joint.settings.stiffness, 4);
        }
      }
      vrm.springBoneManager?.setInitState();
      vrm.springBoneManager?.reset();
      const body = new THREE.Box3().setFromObject(vrm.scene);
      const bonePos = (name: "head" | "chest") =>
        vrm!.humanoid.getRawBoneNode(name)?.getWorldPosition(new THREE.Vector3());
      const head = bonePos("head") ?? body.getCenter(new THREE.Vector3());
      const chest = bonePos("chest")?.y ?? body.max.y - (body.max.y - body.min.y) * 0.35;
      framing = { top: body.max.y, chest, center: new THREE.Vector3(head.x, 0, head.z) };
      frameBody();
      if (vrm.lookAt) vrm.lookAt.target = gazeTarget;
    });

    // The eyes follow this point: the camera (the user's seat), moved towards where the
    // user's face is in their camera
    const gazeTarget = new THREE.Object3D();
    scene.add(gazeTarget);
    const gaze = new THREE.Vector2();
    const cameraRight = new THREE.Vector3();
    const cameraUp = new THREE.Vector3();

    const timer = new THREE.Timer();
    let nextBlink = 2;
    let talk = 0; // 0..1, smoothed speech energy that drives the gestures
    let lastTalkTime = -Infinity;
    let listen = 0; // 0..1, smoothed "the user is talking"
    let nodStart = -Infinity;
    let nextNod = 0;
    const emotionWeights: Record<string, number> = {};
    let frame = 0;
    const animate = (timestamp?: number) => {
      frame = requestAnimationFrame(animate);
      timer.update(timestamp);
      const delta = Math.min(timer.getDelta(), 0.1);
      const time = timer.getElapsed();
      const ease = (rate: number) => 1 - Math.exp(-rate * delta);

      if (vrm && armPose) {
        const expressions = vrm.expressionManager;

        // Lips
        const visemes = lipSyncRef.current?.update(delta);
        let mouth = 0;
        for (const [name, weight] of Object.entries(visemes ?? {})) {
          expressions?.setValue(name, weight);
          mouth += weight;
        }
        talk += (Math.min(1, mouth * 1.5) - talk) * ease(mouth > 0.05 ? 6 : 2);
        if (mouth > 0.05) lastTalkTime = time;

        // Listening: lean in, tilt the head, nod now and then, with an attentive face
        listen += ((userTalkingRef.current ? 1 : 0) - listen) * ease(3);
        if (listen > 0.5 && time > nextNod) {
          nodStart = time;
          nextNod = time + 2.5 + Math.random() * 2;
        }
        const nod = time - nodStart < 0.6 ? Math.sin((Math.PI * (time - nodStart)) / 0.6) * 0.09 : 0;

        // Emotion: blend in, hold while talking and a few seconds after, then back to neutral
        const { name: emotion, at, picked } = emotionRef.current;
        const idleSecs = Math.min(time - lastTalkTime, performance.now() / 1000 - at);
        if (emotion !== "neutral" && !picked && idleSecs > EMOTION_HOLD_SECS) setEmotion("neutral");
        for (const [name, intensity] of Object.entries(EMOTION_INTENSITY)) {
          const attentive = name === "relaxed" && emotion === "neutral" ? listen * 0.25 : 0;
          const target = name === emotion ? intensity : attentive;
          emotionWeights[name] = (emotionWeights[name] ?? 0) + (target - (emotionWeights[name] ?? 0)) * ease(3);
          expressions?.setValue(name, emotionWeights[name]);
        }

        // Blink every 2-6 s, ~150 ms each
        const sinceBlink = time - nextBlink;
        expressions?.setValue("blink", sinceBlink > 0 ? Math.max(0, 1 - Math.abs(sinceBlink - 0.075) / 0.075) : 0);
        if (sinceBlink > 0.15) nextBlink = time + 2 + Math.random() * 4;

        // Gaze: towards the user's face if seen in the last 1.5 s, otherwise straight ahead
        const face = faceRef.current;
        const seen = face && performance.now() - face.at < 1500;
        gaze.x += ((seen ? face.x : 0) - gaze.x) * ease(4);
        gaze.y += ((seen ? face.y : 0) - gaze.y) * ease(4);
        camera.updateMatrixWorld();
        cameraRight.setFromMatrixColumn(camera.matrixWorld, 0);
        cameraUp.setFromMatrixColumn(camera.matrixWorld, 1);
        gazeTarget.position
          .copy(camera.position)
          .addScaledVector(cameraRight, gaze.x * 0.5)
          .addScaledVector(cameraUp, gaze.y * 0.3);

        // Body: breathing and idle sway, nods and arm gestures while talking, listening
        // posture, and the head turning a little towards the user.
        // (VRM 0.x bones are turned around: rotations about X and Z are mirrored)
        const humanoid = vrm.humanoid;
        const mood = emotionWeights;
        const s = vrm.meta.metaVersion === "0" ? -1 : 1;
        humanoid
          .getNormalizedBoneNode("chest")
          ?.rotation.set(
            s * (Math.sin(time * 1.6) * 0.015 + listen * 0.03),
            talk * Math.sin(time * 1.1) * 0.04,
            0,
          );
        humanoid
          .getNormalizedBoneNode("neck")
          ?.rotation.set(
            s *
              (Math.sin(time * 0.7) * 0.03 + talk * Math.sin(time * 2.7) * 0.05 +
                (mood.sad ?? 0) * 0.15 - (mood.surprised ?? 0) * 0.1 +
                listen * 0.04 + nod - gaze.y * 0.08),
            Math.sin(time * 0.45) * 0.06 + talk * Math.sin(time * 1.3) * 0.05 + gaze.x * 0.15,
            s * (Math.sin(time * 0.5) * 0.02 + (mood.happy ?? 0) * 0.07 + listen * 0.05),
          );
        armPose.apply(time, talk);

        vrm.update(delta);
      }
      controls.update(delta);
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("dblclick", resetCamera);
      controls.dispose();
      if (vrm) VRMUtils.deepDispose(vrm.scene);
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [url, setEmotion, faceRef]);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}

type ArmPose = { apply: (time: number, talk: number) => void };

/**
 * Bring the arms down along the body, whatever the rest pose (T-pose, A-pose...),
 * and return a function that adds talking gestures on top of that pose.
 */
function relaxArms(vrm: VRM, degreesBelowHorizontal = 72): ArmPose {
  const bone = vrm.humanoid.getNormalizedBoneNode.bind(vrm.humanoid);
  // VRM 0.x models face -Z (rotateVRM0 turns them around), so their local axes are mirrored
  const sign = vrm.meta.metaVersion === "0" ? -1 : 1;
  vrm.scene.updateMatrixWorld(true);

  const arms = (["left", "right"] as const).flatMap((side) => {
    const direction = side === "left" ? -1 : 1;
    const upper = bone(`${side}UpperArm`);
    const lower = bone(`${side}LowerArm`);
    if (!upper || !lower) return [];
    const arm = lower.getWorldPosition(new THREE.Vector3()).sub(upper.getWorldPosition(new THREE.Vector3()));
    const restAngle = Math.asin(-arm.y / arm.length()); // 0 for a T-pose, ~0.7 rad for an A-pose
    const lowerBy = THREE.MathUtils.degToRad(degreesBelowHorizontal) - restAngle;
    return [{ upper, lower, mirror: direction * sign, lowerBy, phase: side === "left" ? 0 : 1.7 }];
  });

  const apply = (time: number, talk: number) => {
    for (const { upper, lower, mirror, lowerBy, phase } of arms) {
      // Talking: lift the arm a little and bend the elbow, as if gesturing
      const lift = talk * (0.12 + 0.06 * Math.sin(time * 1.9 + phase));
      upper.rotation.set(talk * 0.15, 0, mirror * (lowerBy - lift));
      lower.rotation.set(0, mirror * (0.25 + talk * (0.35 + 0.2 * Math.sin(time * 2.3 + phase))), 0);
    }
  };
  apply(0, 0);
  return { apply };
}
