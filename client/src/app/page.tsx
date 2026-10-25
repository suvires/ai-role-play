"use client";

import dynamic from "next/dynamic";

// The voice UI controls and three.js need the browser (document, WebGL): render client-side only
const AvatarApp = dynamic(() => import("@/components/AvatarApp"), { ssr: false });

export default function Home() {
  return <AvatarApp />;
}
