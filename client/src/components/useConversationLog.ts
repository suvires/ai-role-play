// Records what the user and the bot say, to download the conversation as a text file.

import { useCallback, useState } from "react";
import { RTVIEvent, type BotTTSTextData, type TranscriptData } from "@pipecat-ai/client-js";
import { useRTVIClientEvent } from "@pipecat-ai/client-react";

type Entry = { speaker: "user" | "bot"; text: string; at: Date };

export function useConversationLog() {
  const [entries, setEntries] = useState<Entry[]>([]);

  useRTVIClientEvent(
    RTVIEvent.UserTranscript,
    useCallback((data: TranscriptData) => {
      if (data.final && data.text.trim()) {
        setEntries((log) => [...log, { speaker: "user", text: data.text.trim(), at: new Date() }]);
      }
    }, []),
  );

  // What was actually spoken (no emotion tags), one call per sentence: join a turn's sentences
  useRTVIClientEvent(
    RTVIEvent.BotTtsText,
    useCallback((data: BotTTSTextData) => {
      const text = data.text.trim();
      if (!text) return;
      setEntries((log) => {
        const last = log[log.length - 1];
        if (last?.speaker === "bot") return [...log.slice(0, -1), { ...last, text: `${last.text} ${text}` }];
        return [...log, { speaker: "bot", text, at: new Date() }];
      });
    }, []),
  );

  const clear = useCallback(() => setEntries([]), []);

  const download = useCallback(
    (botName: string) => {
      const time = (date: Date) => date.toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit" });
      const started = entries[0]?.at ?? new Date();
      const lines = entries.map(
        (entry) => `[${time(entry.at)}] ${entry.speaker === "user" ? "Tú" : botName}: ${entry.text}`,
      );
      const header = `Conversación con ${botName} — ${started.toLocaleString("es-ES")}\n\n`;
      const blob = new Blob([header + lines.join("\n\n") + "\n"], { type: "text/plain;charset=utf-8" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `conversacion-${botName.toLowerCase()}-${started.toISOString().slice(0, 16).replace(":", "")}.txt`;
      link.click();
      URL.revokeObjectURL(link.href);
    },
    [entries],
  );

  return { entries, clear, download };
}
