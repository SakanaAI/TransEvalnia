#!/bin/bash
MODEL=qwen
INDIR=sample_data
OUTDIR=sample_data
CORPUS_BASE=wmt2021_en_ja_10
echo Input corpus is "${INDIR}/${CORPUS_BASE}.jsonl"
# First produce the per-translation evaluations.
python3 reason_rank.py \
	--model="${MODEL}" \
	--translation_corpus="${INDIR}/${CORPUS_BASE}.jsonl" \
	--evaluation_corpus="${OUTDIR}/${CORPUS_BASE}_eval_${MODEL}.jsonl" \
	--skip_ranking \
	--permuted_corpus="" \
	--ranked_corpus="" \
	--translation_evaluation_instructions=prompts/translation_evaluation_instructions_generic.txt \
	--ranking_instructions=/dev/null \
	--random_subset_size=-1
# Score the translations
python3 score_translations_main.py \
	--model="${MODEL}" \
	--scored_corpus="${OUTDIR}/${CORPUS_BASE}_eval_scored_${MODEL}.jsonl" \
	--scoring_instructions=prompts/scoring_instructions_generic.txt \
	--evaluation_corpus="${OUTDIR}/${CORPUS_BASE}_eval_${MODEL}.jsonl" \
	--translation_corpus=/dev/null \
	--ranked_corpus=/dev/null \
	--translation_evaluation_instructions=/dev/null
# Run two-step ranking
python3 reason_rank.py \
	--model="${MODEL}" \
	--translation_corpus="${INDIR}/${CORPUS_BASE}.jsonl" \
	--evaluation_corpus="${OUTDIR}/${CORPUS_BASE}_eval_${MODEL}.jsonl" \
	--permuted_corpus="${OUTDIR}/${CORPUS_BASE}_eval_permuted_${MODEL}.jsonl" \
	--ranked_corpus="${OUTDIR}/${CORPUS_BASE}_eval_ranked_${MODEL}.jsonl" \
	--skip_evaluate_translations \
	--translation_evaluation_instructions=prompts/translation_evaluation_instructions_generic.txt \
	--ranking_instructions=prompts/ranking_instructions_generic.txt \
	--random_subset_size=-1 \
	--num_permutations=2
# Run one-step ranking, which needs the permuted data from above.
python3 reason_rank_one_step.py \
	--model="${MODEL}" \
	--translation_corpus="${OUTDIR}/${CORPUS_BASE}_eval_permuted_${MODEL}.jsonl" \
	--ranked_corpus="${OUTDIR}/${CORPUS_BASE}_eval_ranked_one_step_${MODEL}.jsonl" \
	--one_step_ranking_instructions=prompts/ranking_instructions_one_step_generic.txt \
	--evaluation_corpus=/dev/null \
	--translation_evaluation_instructions=/dev/null
# Run three-step (interleaved) ranking
python3 reason_rank.py \
	--model="${MODEL}" \
	--translation_corpus="${INDIR}/${CORPUS_BASE}.jsonl" \
	--evaluation_corpus="${OUTDIR}/${CORPUS_BASE}_eval_${MODEL}.jsonl" \
	--permuted_corpus="${OUTDIR}/${CORPUS_BASE}_eval_permuted_${MODEL}.jsonl" \
	--interleaved_evaluation_corpus="${OUTDIR}/${CORPUS_BASE}_eval_interleaved_${MODEL}.jsonl" \
	--skip_evaluate_translations \
	--skip_permute_corpus \
	--ranked_corpus="${OUTDIR}/${CORPUS_BASE}_eval_interleaved_ranked_${MODEL}.jsonl" \
	--translation_evaluation_instructions=prompts/translation_evaluation_instructions_generic.txt \
	--interleaving_instructions=prompts/interleaving_instructions.txt \
	--interleaved_ranking_instructions=prompts/interleaved_ranking_instructions_generic.txt \
	--random_subset_size=-1 \
	--num_permutations=2
# Run reason ranking without reasoning
python3 reason_rank_one_step.py \
	    --model="${MODEL}" \
	    --translation_corpus="${OUTDIR}/${CORPUS_BASE}_eval_permuted_${MODEL}.jsonl" \
	    --ranked_corpus="${OUTDIR}/${CORPUS_BASE}_ranked_no_reasoning_${MODEL}.jsonl" \
	    --one_step_ranking_instructions=prompts/ranking_instructions_no_reasoning_generic.txt \
	    --evaluation_corpus=/dev/null \
	    --translation_evaluation_instructions=/dev/null
