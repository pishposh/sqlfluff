"""Dataclasses for reflow work."""


from itertools import chain
import logging
from typing import Iterator, List, Optional, Sequence, Tuple, cast
from sqlfluff.core.config import FluffConfig

from sqlfluff.core.parser import BaseSegment, RawSegment
from sqlfluff.core.rules.base import LintFix
from sqlfluff.utils.reflow.config import ReflowConfig
from sqlfluff.utils.reflow.depthmap import DepthMap

from sqlfluff.utils.reflow.elements import ReflowElement, ReflowBlock, ReflowPoint

# We're in the utils module, but users will expect reflow
# logs to appear in the context of rules. Hence it's a subset
# of the rules logger.
reflow_logger = logging.getLogger("sqlfluff.rules.reflow")


class ReflowSequence:
    """Class for keeping track of elements in a reflow operation.

    It is assumed that there will be alternating blocks and points
    (even if some points have no segments). This is validated on
    construction.

    We assume points on each end because this is the case with a file.
    """

    def __init__(
        self,
        elements: Sequence[ReflowElement],
        root_segment: BaseSegment,
        reflow_config: ReflowConfig,
        depth_map: DepthMap,
        embodied_fixes: Optional[List[LintFix]] = None,
    ):
        # First validate integrity
        self._validate_reflow_sequence(elements)
        # Then save
        self.elements = elements
        self.root_segment = root_segment
        self.reflow_config = reflow_config
        self.depth_map = depth_map
        # This keeps track of fixes generated in the chaining process.
        # Alternatively pictured: This is the list of fixes required
        # to generate this sequence. We can build on this as we edit
        # the sequence.
        self.embodied_fixes: List[LintFix] = embodied_fixes or []

    def get_fixes(self):
        """Get the current fix buffer."""
        return self.embodied_fixes

    @staticmethod
    def _validate_reflow_sequence(elements: Sequence[ReflowElement]):
        assert elements, "ReflowSequence has empty elements."
        # Check odds and events
        OddType = elements[0].__class__
        EvenType = ReflowPoint if OddType is ReflowBlock else ReflowBlock
        try:
            # Check odds are all points
            assert all(
                isinstance(elem, OddType) for elem in elements[::2]
            ), f"Not all odd elements are {OddType.__name__}"
            # Check evens are all blocks
            assert all(
                isinstance(elem, EvenType) for elem in elements[1::2]
            ), f"Not all even elements are {EvenType.__name__}"
        except AssertionError as err:  # pragma: no cover
            for elem in elements:
                reflow_logger.error("   - %s", elem)
            reflow_logger.exception("Assertion check on ReflowSequence failed.")
            raise err

    @staticmethod
    def _elements_from_raw_segments(
        segments: Sequence[RawSegment], reflow_config: ReflowConfig, depth_map: DepthMap
    ) -> Sequence[ReflowElement]:
        """Construct reflow elements from raw segments.

        NOTE: ReflowBlock elements should only ever have one segment
        which simplifies iteration here.
        """
        elem_buff: List[ReflowElement] = []
        seg_buff: List[RawSegment] = []
        for seg in segments:
            if seg.is_type("whitespace", "newline", "end_of_file", "indent"):
                # Add to the buffer and move on.
                seg_buff.append(seg)
                continue
            elif elem_buff or seg_buff:
                # There are elements. The last will have been a block.
                # Add a point before we add the block. NOTE: It may be empty.
                elem_buff.append(ReflowPoint(segments=seg_buff))
            # Add the block, with config info.
            elem_buff.append(
                ReflowBlock.from_config(
                    segments=[seg],
                    config=reflow_config,
                    depth_info=depth_map.get_depth_info(seg),
                )
            )
            # Empty the buffer
            seg_buff = []

        # If we ended with a buffer, apply it.
        if seg_buff:
            elem_buff.append(ReflowPoint(segments=seg_buff))
        return elem_buff

    @classmethod
    def from_raw_segments(
        cls,
        segments: Sequence[RawSegment],
        root_segment: BaseSegment,
        config: FluffConfig,
        depth_map: Optional[DepthMap] = None,
    ):
        """Construct a ReflowSequence from a sequence of raw segments.

        Aimed to be the basic constructor, which other more specific
        ones may fall back to.
        """
        reflow_config = ReflowConfig.from_fluff_config(config)
        if depth_map is None:
            depth_map = DepthMap.from_raws_and_root(segments, root_segment)
        return cls(
            elements=cls._elements_from_raw_segments(
                segments,
                reflow_config=reflow_config,
                # NOTE: This pathway is inefficient. Ideally the depth
                # map should be constructed elsewhere and then passed in.
                depth_map=depth_map,
            ),
            root_segment=root_segment,
            reflow_config=reflow_config,
            depth_map=depth_map,
        )

    @classmethod
    def from_root(cls, root_segment: BaseSegment, config: FluffConfig):
        """Generate a sequence from a root segment."""
        return cls.from_raw_segments(
            root_segment.raw_segments,
            root_segment,
            config=config,
            # This is the efficient route. We use it here because we can.
            depth_map=DepthMap.from_parent(root_segment),
        )

    @classmethod
    def from_around_target(
        cls,
        target_segment: BaseSegment,
        root_segment: BaseSegment,
        config: FluffConfig,
        sides: str = "both",
    ):
        """Generate a sequence around a target.

        Args:
            target_segment (:obj:`RawSegment`): The segment to center
                around when considering the sequence to construct.
            root_segment (:obj:`BaseSegment`): The relevant root
                segment (usually the base :obj:`FileSegment`).
            config (:obj:`FluffConfig`): A config object from which
                to load the spacing behaviours of different segments.
            sides (:obj:`str`): Limit the reflow sequence to just one
                side of the target. Default is two sided ("both"), but
                set to "before" or "after" to limit to either side.


        To evaluate reflow around a specific target, we need
        need to generate a sequence which goes for the preceding
        raw to the following raw.
        i.e. block - point - block - point - block
        (where the central block is the target).
        """
        # There's probably a more efficient way than immediately
        # materialising the raw_segments for the whole root, but
        # it works. Optimise later.
        all_raws = root_segment.raw_segments

        target_raws = target_segment.raw_segments
        assert target_raws
        pre_idx = all_raws.index(target_raws[0])
        post_idx = all_raws.index(target_raws[-1]) + 1
        initial_idx = (pre_idx, post_idx)
        if sides in ("both", "before"):
            # Catch at least the previous segment
            pre_idx -= 1
            while pre_idx - 1 > 0 and all_raws[pre_idx].is_type(
                "whitespace", "newline", "indent"
            ):
                pre_idx -= 1
        if sides in ("both", "after"):
            while post_idx < len(all_raws) and all_raws[post_idx].is_type(
                "whitespace", "newline", "indent"
            ):
                post_idx += 1
            # Capture one more after the whitespace.
            post_idx += 1
        segments = all_raws[pre_idx:post_idx]
        reflow_logger.debug(
            "Generating ReflowSequence.from_around_target(). idx: %s. "
            "slice: %s:%s. segments: %s",
            initial_idx,
            pre_idx,
            post_idx,
            segments,
        )
        return cls.from_raw_segments(segments, root_segment, config=config)

    def _find_element_idx_with(self, target: RawSegment) -> int:
        for idx, elem in enumerate(self.elements):
            if target in elem.segments:
                return idx
        raise ValueError(  # pragma: no cover
            f"Target [{target}] not found in ReflowSequence."
        )

    def without(self, target: RawSegment) -> "ReflowSequence":
        """Returns a new reflow sequence without the specified segment.

        It's important to note that this doesn't itself remove the target
        from the file. This just allows us to simulate a sequence without it
        and work out what additional whitespace changes would be required
        if we were to remove it.
        """
        removal_idx = self._find_element_idx_with(target)
        if removal_idx == 0 or removal_idx == len(self.elements) - 1:
            raise NotImplementedError(  # pragma: no cover
                "Unexpected removal at one end of a ReflowSequence."
            )
        if isinstance(self.elements[removal_idx], ReflowPoint):
            raise NotImplementedError(  # pragma: no cover
                "Not expected removal of whitespace in ReflowSequence."
            )
        merged_point = ReflowPoint(
            segments=list(self.elements[removal_idx - 1].segments)
            + list(self.elements[removal_idx + 1].segments),
        )
        return ReflowSequence(
            elements=list(self.elements[: removal_idx - 1])
            + [merged_point]
            + list(self.elements[removal_idx + 2 :]),
            root_segment=self.root_segment,
            reflow_config=self.reflow_config,
            depth_map=self.depth_map,
            # Generate the fix to do the removal.
            embodied_fixes=[LintFix.delete(target)],
        )

    def insert(
        self, insertion: RawSegment, target: RawSegment, pos="before"
    ) -> "ReflowSequence":
        """Returns a new reflow sequence with the new element inserted.

        Insertion is always relative to an existing element. Either before
        or after it as specified by `pos`.
        """
        assert pos in ("before", "after")
        target_idx = self._find_element_idx_with(target)
        # Are we trying to insert something whitespace-like?
        if insertion.is_type("whitespace", "indent", "newline"):  # pragma: no cover
            raise ValueError(
                "ReflowSequence.insert() does not support direct insertion of "
                "spacing elements such as whitespace or newlines"
            )

        # We're inserting something blocky. That means a new block AND a new point.
        # It's possible we try to _split_ a point by targetting a whitespace element
        # inside a larger point. For now this isn't supported.
        # NOTE: We use the depth info of the reference anchor, with the assumption
        # (I think reliable) that the insertion will be applied as a sibling of
        # the target.
        self.depth_map.copy_depth_info(target, insertion)
        new_block = ReflowBlock.from_config(
            segments=[insertion],
            config=self.reflow_config,
            depth_info=self.depth_map.get_depth_info(target),
        )
        if isinstance(self.elements[target_idx], ReflowPoint):
            raise NotImplementedError(  # pragma: no cover
                "Can't insert relative to whitespace for now."
            )
        elif pos == "before":
            return ReflowSequence(
                elements=list(self.elements[:target_idx])
                + [new_block, ReflowPoint([])]
                + list(self.elements[target_idx:]),
                root_segment=self.root_segment,
                reflow_config=self.reflow_config,
                depth_map=self.depth_map,
                # Generate the fix to do the removal.
                embodied_fixes=[LintFix.create_before(target, [insertion])],
            )
        elif pos == "after":  # pragma: no cover
            # TODO: This doesn't get coverage - should it even exist?
            # Re-evaluate whether this code path is ever taken once more rules use
            # this.
            return ReflowSequence(
                elements=list(self.elements[: target_idx + 1])
                + [ReflowPoint([]), new_block]
                + list(self.elements[target_idx + 1 :]),
                root_segment=self.root_segment,
                reflow_config=self.reflow_config,
                depth_map=self.depth_map,
                # Generate the fix to do the removal.
                embodied_fixes=[LintFix.create_after(target, [insertion])],
            )
        raise ValueError(
            f"Unexpected value for ReflowSequence.insert(pos): {pos}"
        )  # pragma: no cover

    def replace(
        self, target: BaseSegment, edit: Sequence[BaseSegment]
    ) -> "ReflowSequence":
        """Returns a new reflow sequence elements replaced."""
        replace_fix = LintFix.replace(target, edit)

        target_raws = target.raw_segments
        assert target_raws

        edit_raws = list(chain.from_iterable(seg.raw_segments for seg in edit))

        # Add the new segments to the depth map at the same level as the target.
        # First work out how much to trim by.
        trim_amount = len(target.path_to(target_raws[0])) - 1
        reflow_logger.debug(
            "Replacement trim amount: %s.",
            trim_amount,
        )
        for edit_raw in edit_raws:
            # NOTE: if target raws has more than one segment we take the depth info
            # of the first one. We trim to avoid including the implications of removed
            # "container" segments.
            self.depth_map.copy_depth_info(target_raws[0], edit_raw, trim=trim_amount)

        # It's much easier to just totally reconstruct the sequence rather
        # than do surgery on the elements.

        # TODO: The surgery is actually a good idea for long sequences now that
        # we have the handle the depth map.

        current_raws = list(
            chain.from_iterable(elem.segments for elem in self.elements)
        )
        start_idx = current_raws.index(target_raws[0])
        last_idx = current_raws.index(target_raws[-1])

        return ReflowSequence(
            self._elements_from_raw_segments(
                current_raws[:start_idx] + edit_raws + current_raws[last_idx + 1 :],
                reflow_config=self.reflow_config,
                # NOTE: the depth map has been mutated to include the new segments.
                depth_map=self.depth_map,
            ),
            root_segment=self.root_segment,
            reflow_config=self.reflow_config,
            depth_map=self.depth_map,
            embodied_fixes=[replace_fix],
        )

    def _iter_points_with_constraints(
        self,
    ) -> Iterator[Tuple[ReflowPoint, Optional[ReflowBlock], Optional[ReflowBlock]]]:
        for idx, elem in enumerate(self.elements):
            # Only evaluate points.
            if isinstance(elem, ReflowPoint):
                pre = None
                post = None
                if idx > 0:
                    pre = cast(ReflowBlock, self.elements[idx - 1])
                if idx < len(self.elements) - 1:
                    post = cast(ReflowBlock, self.elements[idx + 1])
                yield elem, pre, post

    def respace(self, strip_newlines=False, filter="all") -> "ReflowSequence":
        """Respace a sequence.

        Args:
            strip_newlines (:obj:`bool`): Optionally strip newlines
                before respacing. This is primarily used on focussed
                sequences to coerce objects onto a single line. This
                does not apply any prioritisation to which line breaks
                to remove and so is not a substitute for the full
                `reindent` or `reflow` methods.
            filter (:obj:`str`): Optionally filter which reflow points
                to respace. Default configuration is `all`. Other options
                are `line_break` which only respaces points containing
                a `newline` or `end_of_file` marker or `inline` which
                is the inverse of `line_break`. This is most useful for
                filtering between trailing whitespace and fixes between
                content on a line.

        This resets spacing in a ReflowSequence. Note, it relies on the
        embodied fixes being correct so that we can build on them.
        """
        assert filter in (
            "all",
            "newline",
            "inline",
        ), f"Unexpected value for filter: {filter}"
        # Use the embodied fixes as a starting point.
        fixes = self.embodied_fixes or []
        new_elements: List[ReflowElement] = []
        for point, pre, post in self._iter_points_with_constraints():
            # We filter on the elements POST RESPACE. This is to allow
            # strict respacing to reclaim newlines.
            new_fixes, new_point = point.respace_point(
                prev_block=pre,
                next_block=post,
                fixes=fixes,
                strip_newlines=strip_newlines,
            )
            # If filter has been set, optionally unset the returned values.
            if (
                filter == "inline"
                if (
                    # NOTE: We test on the NEW point.
                    any(
                        seg.is_type("newline", "end_of_file")
                        for seg in new_point.segments
                    )
                )
                else filter == "newline"
            ):
                # Reset the values
                new_point = point
            # Otherwise apply the new fixes
            else:
                fixes = new_fixes

            if pre and (not new_elements or new_elements[-1] != pre):
                new_elements.append(pre)
            new_elements.append(new_point)
            if post:
                new_elements.append(post)
        return ReflowSequence(
            elements=new_elements,
            root_segment=self.root_segment,
            reflow_config=self.reflow_config,
            depth_map=self.depth_map,
            # Generate the fix to do the removal.
            embodied_fixes=fixes,
        )
