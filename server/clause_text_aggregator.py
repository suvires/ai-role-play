#
# Text aggregator that releases the first clause of each response early
#

import re
from typing import Optional

from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

# A comma, semicolon or colon followed by whitespace (so "3,5" is never split)
_CLAUSE_END = re.compile(r"[,;:](?=\s)")


class FirstClauseTextAggregator(SimpleTextAggregator):
    """Sentence aggregator that also splits the *first* chunk of a response at a comma.

    TTS can't start until it has a full chunk, so waiting for the end of a long first
    sentence delays the bot's first words. Only the first chunk is split: the rest are
    generated while the bot is already speaking, and full sentences sound more natural.
    """

    def __init__(self, min_words: int = 4):
        super().__init__()
        self._min_words = min_words
        self._first_chunk = True

    async def aggregate(self, text: str) -> Optional[str]:
        result = await super().aggregate(text)
        if result is None and "\n" in self._text.strip():
            # Line breaks end a chunk too (e.g. poem verses, which often lack punctuation)
            line, _, rest = self._text.lstrip().partition("\n")
            result, self._text = line, rest
        if result is None and self._first_chunk:
            for match in _CLAUSE_END.finditer(self._text):
                clause = self._text[: match.end()]
                if len(clause.split()) >= self._min_words:
                    result = clause
                    self._text = self._text[match.end() :]
                    break
        if result is not None:
            self._first_chunk = False
        return result

    async def handle_interruption(self):
        await super().handle_interruption()
        self._first_chunk = True

    async def reset(self):
        await super().reset()
        self._first_chunk = True
