"""No promotion from training scores, selected seeds, or partial relay results."""
import math


def promotion_gate(training_seeds,holdout_results,expected_kind,minimum_runs=4):
    reasons=[];seeds=[r['seed'] for r in holdout_results]
    if len(seeds)<minimum_runs:reasons.append('insufficient_holdout_runs')
    if len(set(seeds))!=len(seeds):reasons.append('duplicate_holdout_seed')
    if set(training_seeds)&set(seeds):reasons.append('train_holdout_overlap')
    for r in holdout_results:
        if r.get('kind')!=expected_kind:reasons.append('wrong_task')
        if r.get('error') or r.get('passed') is not True:reasons.append('failed_holdout')
        if not r.get('evidence_sha256') or not r.get('protocol_sha256'):reasons.append('missing_provenance')
    if len({r.get('protocol_sha256') for r in holdout_results})>1:reasons.append('mixed_protocols')
    return dict(promoted=not reasons,reasons=sorted(set(reasons)),seeds=seeds)
