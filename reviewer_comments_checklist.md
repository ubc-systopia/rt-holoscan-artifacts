# Reviewer Comments Checklist

- [X] Clarify that all Holoscan operators share a context being scheduled; explain that this is intra-task scheduling and differs from related work.

- [X] Clarify “reconfiguration is performed at runtime” in Sec. II-C: Green Contexts cannot be resized after creation during an application run.

- [X] Rewrite the first paragraph of “SM partition selection” to be denser and less AI-sounding.

- [X] Improve clarity from “We write $o^i$ to denote…” through the end of the Sec. IV-A paragraph.

- [U] Remove or revise the redundant second half of the third sentence in the final paragraph of Sec. IV-A.

- [ ] Condense Sec. IV-C; state whether the transformation preserves the invariants required by prior analysis.

- [U] Before Sec. V, make clear that GC partitions are statically set per operator.

- [X] Resolve the conflicting meanings of $|P|$ in Algorithm 1 and Equation 1.

- [X] Remove spaces around em dashes in paragraph 2 of Sec. V.

- [ ] Justify the Sec. V claim about integer programming, particularly given smooth execution-time/SM-count relationships.

- [X] Clarify the representation of $p$ in paragraph 4 of Sec. V.

- [X] Move or omit unnecessary detail in Sec. V paragraph 4, potentially placing it in an algorithm figure.

- [X] Define “balanced-transfer neighborhood” in the final paragraph of Sec. V.

- [X] In Sec. VI, explicitly state that the embedded GPU on the Orin is not used.

- [X] Remove or rewrite “is not an oversight” in Sec. VI-A, paragraph 3.

- [X] Explain polling before it appears in the final paragraph of Sec. VI-A; clarify whether the greedy scheduler polls and what it is compared against.

- [U] Define the “compute” method in Sec. VI-B, paragraph 2.

- [ ] Clarify what “4 random seeds” refers to in Sec. VI-B, paragraph 3.

- [ ] Explain why sample counts are approximate rather than exactly 4,000.

- [ ] Consider citing Liu, Wagle, Anderson, Yang, Zhang, and Li (ECRTS 2024) for CPU-side CUDA contention in Sec. VI-B.

- [ ] Measure CPU-side interference from asynchronous memory copies, as identified in that work.

- [X] Reassess or support the claim that GC introduces additional host-side overhead in Sec. VI-B.

- [ ] Reconcile the Sec. II-B statement that CPU interference is not considered with the CPU-interference discussion in Sec. VI-B.

- [ ] Clarify which Sec. III overhead measurement is referenced by the first sentence of the fourth paragraph of Sec. VI-D.
