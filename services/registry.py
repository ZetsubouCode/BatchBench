from .webp_converter import handle as webp_handle
from .batch_adjust import handle as batch_handle
from .tag_editor import handle as tags_handle
from .combine_datasets import handle as combine_handle
from .merge_groups_tool import handle as merge_groups_handle
from .group_renamer import handle as renamer_handle
from .webtoon_splitter import handle as webtoon_handle
from .offline_tagger import handle as offline_tagger_handle

TOOL_REGISTRY = {
    "webp": webp_handle,
    "batch": batch_handle,
    "tags": tags_handle,
    "combine": combine_handle,
    "merge_groups": merge_groups_handle,
    "rename": renamer_handle,
    "webtoon": webtoon_handle,
    "offline_tagger": offline_tagger_handle,
}
