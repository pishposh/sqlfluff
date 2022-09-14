"""Dataclasses for reflow work."""


from itertools import chain
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, cast

from sqlfluff.core.parser import BaseSegment, RawSegment
from sqlfluff.core.parser.segments.raw import NewlineSegment, WhitespaceSegment
from sqlfluff.core.rules.base import LintFix

from sqlfluff.utils.reflow.config import ReflowConfig
from sqlfluff.utils.reflow.depthmap import DepthInfo

# We're in the utils module, but users will expect reflow
# logs to appear in the context of rules. Hence it's a subset
# of the rules logger.
reflow_logger = logging.getLogger("sqlfluff.rules.reflow")


@dataclass(frozen=True)
class ReflowElement:
    """Base reflow element class."""

    segments: Sequence[RawSegment]

    @staticmethod
    def _class_types(segments: Sequence[RawSegment]) -> Set[str]:
        return set(chain.from_iterable(seg.class_types for seg in segments))

    @property
    def class_types(self):
        """The set of contained class types.

        Parallel to BaseSegment.class_types
        """
        return self._class_types(self.segments)

    def num_newlines(self) -> int:
        """How many newlines does this element contain?"""
        return sum(
            bool(seg.class_types.intersection({"newline", "end_of_file"}))
            for seg in self.segments
        )


@dataclass(frozen=True)
class ReflowBlock(ReflowElement):
    """Class for keeping track of elements to reflow.

    It holds segments to reflow and also exposes configuration
    around how they are expected to reflow around others.

    The attributes exposed are designed to be "post configuration"
    i.e. they should reflect configuration appropriately.

    NOTE: These are the smallest unit of "work" within
    the reflow methods, and may contain meta segments.
    """

    # Options for spacing rules are:
    # - single:         the default (one single space)
    # - touch:          no whitespace
    # - any:            don't enforce any spacing rules
    spacing_before: str
    spacing_after: str
    # - None:           the default (no particular preference)
    # - leading:        prefer newline before
    # - trailing:       prefer newline after
    line_position: Optional[str]
    # The depth info is used in determining where to put line breaks.
    depth_info: DepthInfo
    # This stores relevant configs for segments in the stack.
    stack_spacing_configs: Dict[int, str]
    line_position_configs: Dict[int, str]

    @classmethod
    def from_config(cls, segments, config: ReflowConfig, depth_info: DepthInfo):
        """Extendable constructor which accepts config."""
        block_config = config.get_block_config(cls._class_types(segments))
        # Populate any spacing_within config.
        # TODO: This needs decent unit tests - not just what happens in rules.
        stack_spacing_configs = {}
        line_position_configs = {}
        for hash, class_types in zip(
            depth_info.stack_hashes, depth_info.stack_class_types
        ):
            spacing_within = config.get_block_config(class_types).get(
                "spacing_within", None
            )
            line_position = config.get_block_config(class_types).get(
                "line_position", None
            )
            if spacing_within:
                stack_spacing_configs[hash] = spacing_within
            if line_position:
                line_position_configs[hash] = line_position
        return cls(
            segments=segments,
            spacing_before=block_config.get("spacing_before", "single"),
            spacing_after=block_config.get("spacing_after", "single"),
            line_position=block_config.get("line_position", None),
            depth_info=depth_info,
            stack_spacing_configs=stack_spacing_configs,
            line_position_configs=line_position_configs,
        )


@dataclass(frozen=True)
class ReflowPoint(ReflowElement):
    """Class for keeping track of editable elements in reflow.

    It holds segments which can be changed during a reflow operation
    such as whitespace and newlines.

    It holds no configuration and is influenced by the blocks either
    side.
    """

    def _get_indent_segment(self) -> Optional[RawSegment]:
        """Get the current indent segment (if there)."""
        for seg in reversed(self.segments):
            if seg.is_type("newline", "end_of_file"):
                return None
            elif seg.is_type("whitespace"):
                return seg
        return None

    def get_indent(self) -> Optional[str]:
        """Get the current indent (if there)."""
        seg = self._get_indent_segment()
        return seg.raw if seg else None

    def indent_to(
        self,
        desired_indent: str,
        after: Optional[BaseSegment] = None,
        before: Optional[BaseSegment] = None,
    ) -> Tuple[List[LintFix], "ReflowPoint"]:
        """Coerce a point to have a particular indent.

        If the point currently contains no newlines, one will
        be introduced and any trailing whitespace will be effectively
        removed.

        More specifically, the newline is _inserted_ before the existing
        whitespace, with the new indent being a replacement for that
        same whitespace.
        """
        # Get the indent (or in the case of no newline, the last whitespace)
        indent_seg = self._get_indent_segment()
        if self.num_newlines():
            # There is already a newline.
            if indent_seg:
                # Coerce existing indent to desired.
                if indent_seg.raw == desired_indent:
                    # trivial case. indent already correct
                    return [], self
                new_indent = indent_seg.edit(desired_indent)
                idx = self.segments.index(indent_seg)
                return [LintFix.replace(indent_seg, [new_indent])], ReflowPoint(
                    list(self.segments[:idx])
                    + [new_indent]
                    + list(self.segments[idx + 1 :])
                )
            else:
                # There is a newline, but no indent. Make one after the newline
                idx = max(
                    i for i, seg in enumerate(self.segments) if seg.is_type("newline")
                )
                new_indent = WhitespaceSegment(desired_indent)
                return [
                    LintFix.create_after(self.segments[idx], [new_indent])
                ], ReflowPoint(
                    list(self.segments[: idx + 1])
                    + [new_indent]
                    + list(self.segments[idx + 1 :])
                )
        else:
            # There isn't currently a newline.
            new_newline = NewlineSegment()
            if not indent_seg:
                # There isn't a whitespace segment either. We need to insert one.
                # Do we have an anchor?
                new_indent = WhitespaceSegment(desired_indent)
                if not before and not after:
                    raise NotImplementedError(
                        "Not set up to handle empty points in this "
                        "scenario without provided before/after "
                        f"anchor: {self.segments}"
                    )
                # Otherwise make a new indent, attached to the relevant anchor.
                elif before:
                    fix = LintFix.create_before(before, [new_newline, new_indent])
                else:
                    assert after  # mypy hint
                    fix = LintFix.create_after(after, [new_newline, new_indent])
                new_point = ReflowPoint([new_newline, new_indent])
            else:
                # There is whitespace. Coerce it to the right indent and add
                # a newline _before_
                idx = self.segments.index(indent_seg)
                new_indent = indent_seg.edit(desired_indent)
                fix = LintFix.replace(indent_seg, [new_newline, new_indent])
                new_point = ReflowPoint(
                    list(self.segments[:idx])
                    + [new_newline, new_indent]
                    + list(self.segments[idx + 1 :])
                )

            return [fix], new_point

    def respace_point(
        self,
        prev_block: Optional[ReflowBlock] = None,
        next_block: Optional[ReflowBlock] = None,
        fixes: Optional[List[LintFix]] = None,
        strip_newlines: bool = False,
        anchor_on: str = "before",
    ) -> Tuple[List[LintFix], "ReflowPoint"]:
        """Respace a point based on given constraints.

        NB: This effectively includes trailing whitespace fixes.

        Deletion and edit fixes are generated immediately, but creations
        are paused to the end and done in bulk so as not to generate conflicts.

        Note that the `strip_newlines` functionality exists here as a slight
        exception to pure respacing, but as a very simple case of positioning
        line breaks. The default operation of `respace` does not enable it
        however it exists as a convenience for rules which wish to use it.
        """
        new_fixes = []
        last_whitespace: List[RawSegment] = []
        # The buffer is used to create the new reflow point to return
        segment_buffer = list(self.segments)
        edited = False
        pre_constraint = prev_block.spacing_after if prev_block else "single"
        post_constraint = next_block.spacing_before if next_block else "single"

        # Work out the common parent segment and depth
        if prev_block and next_block:
            common = prev_block.depth_info.common_with(next_block.depth_info)
            # Just check the most immediate parent for now for speed.
            # TODO: Review whether this is enough.
            # NOTE: spacing configs will be available on both sides if they're common
            # so it doesn't matter whether we get it from prev_block or next_block.
            within_constraint = prev_block.stack_spacing_configs.get(common[-1], None)
            if not within_constraint:
                pass
            elif within_constraint in ("touch", "inline"):
                # NOTE: inline is actually a more extreme version of "touch".
                # i.e. inline, implies no spaces between either.
                if within_constraint == "inline":
                    # If they are then strip newlines.
                    strip_newlines = True
                # If segments are expected to be touch within. Then modify
                # constraints accordingly.
                # NOTE: We don't override if it's already "any"
                if pre_constraint != "any":
                    pre_constraint = "touch"
                if post_constraint != "any":
                    post_constraint = "touch"
            else:  # pragma: no cover
                idx = prev_block.depth_info.stack_hashes.index(common[-1])
                raise NotImplementedError(
                    f"Unexpected within constraint: {within_constraint} for "
                    f"{prev_block.depth_info.stack_class_types[idx]}"
                )

        reflow_logger.debug("Respacing: %s", self)
        for idx, seg in enumerate(self.segments):
            # If it's whitespace, store it.
            if seg.is_type("whitespace"):
                last_whitespace.append(seg)
            # If it's a newline, react accordingly.
            elif seg.is_type("newline", "end_of_file"):
                # Are we stripping newlines?
                if strip_newlines and seg.is_type("newline"):
                    reflow_logger.debug("    Stripping newline: %s", seg)
                    # Generate a fix to remove it.
                    new_fixes.append(LintFix("delete", seg))
                    # Remove it from the buffer.
                    segment_buffer.remove(seg)
                    # Carry on as though it wasn't here.
                    continue

                # Check if we've just passed whitespace. If we have, remove it
                # as trailing whitespace, both from the buffer and create a fix.
                if last_whitespace:
                    for ws in last_whitespace:
                        segment_buffer.remove(ws)
                        new_fixes.append(LintFix("delete", ws))
                    reflow_logger.debug("    Removing trailing whitespace.")
                # Regardless, unset last_whitespace.
                # We either just deleted it, or it's not relevant for any future
                # segments.
                last_whitespace = []

        if len(last_whitespace) >= 2:
            reflow_logger.debug("   Removing adjoining whitespace.")
            # If we find multiple sequential whitespaces, it's the sign
            # that we've removed something. Only the first one should be
            # a valid indent (or the one we consider for constraints).
            # Remove all the following ones.
            for ws in last_whitespace[1:]:
                segment_buffer.remove(ws)
                new_fixes.append(LintFix("delete", ws))

        # Is there a newline? (that we've not already stripped!)
        if self.num_newlines() and not strip_newlines:
            # Most of this section should be handled as _Indentation_.
            # BUT: There is one case we should handle here.
            # If we find that the last whitespace has a newline
            # before it, and the position markers imply there was
            # a removal between them. Remove the whitespace.
            # This ensures a consistent indent.
            # TODO: Check this doesn't duplicate indentation code
            # once written.

            # The test is less about whether it's longer than one
            # (because we should already have removed additional
            # whitespace above). This is about making it there is
            # at least SOME.
            if len(last_whitespace) == 1:
                ws_seg = last_whitespace[0]
                ws_idx = self.segments.index(ws_seg)
                if ws_idx > 0:
                    prev_seg = self.segments[ws_idx - 1]
                    if (
                        prev_seg.is_type("newline")
                        # Not just unequal. Must be actively _before_.
                        # NOTE: Based on working locations
                        and prev_seg.get_end_loc() < ws_seg.get_start_loc()
                    ):
                        reflow_logger.debug(
                            "    Removing non-contiguous whitespace post removal."
                        )
                        segment_buffer.remove(ws_seg)
                        new_fixes.append(LintFix("delete", ws_seg))

        # Is this an inline case? (i.e. no newline)
        else:
            reflow_logger.debug(
                "    Inline case. Constraints: %s <-> %s.",
                pre_constraint,
                post_constraint,
            )

            # Do we at least have _some_ whitespace?
            if last_whitespace:
                # We do - is it the right size?
                ws_seg = last_whitespace[0]
                ws_idx = segment_buffer.index(ws_seg)

                # Do we have either side set to "any"
                if "any" in [pre_constraint, post_constraint]:
                    # In this instance - don't change anything.
                    # e.g. this could mean there is a comment on one side.
                    pass

                # Do we have either side set to "touch"?
                elif "touch" in [pre_constraint, post_constraint]:
                    # In this instance - no whitespace is correct, This
                    # means we should delete it.
                    new_fixes.append(
                        LintFix(
                            "delete",
                            anchor=ws_seg,
                        )
                    )
                    segment_buffer.pop(ws_idx)

                # Handle the default case
                elif pre_constraint == post_constraint == "single":
                    if ws_seg.raw != " ":
                        new_seg = ws_seg.edit(" ")
                        new_fixes.append(
                            LintFix(
                                "replace",
                                anchor=ws_seg,
                                edit=[new_seg],
                            )
                        )
                        segment_buffer[ws_idx] = new_seg

                else:
                    raise NotImplementedError(  # pragma: no cover
                        f"Unexpected Constraints: {pre_constraint}, {post_constraint}"
                    )
            else:
                # No. Should we insert some?

                # Do we have either side set to "touch" or "any"
                if {"touch", "any"}.intersection([pre_constraint, post_constraint]):
                    # In this instance - no whitespace is correct.
                    # Either because there shouldn't be, or because "any"
                    # means we shouldn't check.
                    pass
                # Handle the default case
                elif pre_constraint == post_constraint == "single":
                    # Insert a single whitespace.
                    reflow_logger.debug("    Inserting Single Whitespace.")
                    # Add it to the buffer first (the easy bit)
                    segment_buffer = [WhitespaceSegment()]

                    # So special handling here. If segments either side
                    # already exist then we don't care which we anchor on
                    # but if one is already an insertion (as shown by a lack)
                    # of pos_marker, then we should piggy back on that pre-existing
                    # fix.
                    existing_fix = None
                    insertion = None
                    if prev_block and not prev_block.segments[-1].pos_marker:
                        existing_fix = "after"
                        insertion = prev_block.segments[-1]
                    elif next_block and not next_block.segments[0].pos_marker:
                        existing_fix = "before"
                        insertion = next_block.segments[0]

                    if existing_fix:
                        reflow_logger.debug(
                            "    Detected existing fix %s", existing_fix
                        )
                        if not fixes:  # pragma: no cover
                            raise ValueError(
                                "Fixes detected, but none passed to .respace(). "
                                "This will cause conflicts."
                            )
                        # Find the fix
                        for fix in fixes:
                            # Does it contain the insertion?
                            # TODO: This feels ugly - eq for BaseSegment is different
                            # to uuid matching for RawSegment. Perhaps this should be
                            # more aligned. There might be a better way of doing this.
                            if (
                                insertion
                                and fix.edit
                                and insertion.uuid in [elem.uuid for elem in fix.edit]
                            ):
                                break
                        else:  # pragma: no cover
                            reflow_logger.warning("Fixes %s", fixes)
                            raise ValueError(f"Couldn't find insertion for {insertion}")
                        # Mutate the existing fix
                        assert fix
                        assert (
                            fix.edit
                        )  # It's going to be an edit if we've picked it up.
                        if existing_fix == "before":
                            fix.edit = [
                                cast(BaseSegment, WhitespaceSegment())
                            ] + fix.edit
                        elif existing_fix == "after":
                            fix.edit = fix.edit + [
                                cast(BaseSegment, WhitespaceSegment())
                            ]
                        edited = True
                    else:
                        reflow_logger.debug(
                            "    Not Detected existing fix. Creating new"
                        )
                        # Take into account hint on where to anchor if given.
                        if prev_block and anchor_on != "after":
                            new_fixes.append(
                                LintFix(
                                    "create_after",
                                    anchor=prev_block.segments[-1],
                                    edit=[WhitespaceSegment()],
                                )
                            )
                        # TODO: We should have a test which covers this clause.
                        elif next_block:  # pragma: no cover
                            new_fixes.append(
                                LintFix(
                                    "create_before",
                                    anchor=next_block.segments[0],
                                    edit=[WhitespaceSegment()],
                                )
                            )
                        else:  # pragma: no cover
                            NotImplementedError(
                                "Not set up to handle a missing _after_ and _before_."
                            )
                else:  # pragma: no cover
                    # TODO: This will get test coverage when configuration routines
                    # are in properly.
                    raise NotImplementedError(
                        f"Unexpected Constraints: {pre_constraint}, {post_constraint}"
                    )

        # Only log if we actually made a change.
        if new_fixes or edited:
            reflow_logger.debug(
                "    Fixes. Old & Changed: %s. New: %s", fixes, new_fixes
            )
        return (fixes or []) + new_fixes, ReflowPoint(segment_buffer)
