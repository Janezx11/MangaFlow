from enum import Enum


class ComicStyle(str, Enum):
    WESTERN_COMIC = "western_comic"
    MANGA = "manga"
    WEBTOON = "webtoon"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
