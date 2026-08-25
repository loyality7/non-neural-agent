"""Entry point: run all experiments. python run.py"""

from experiments import exp0_entity_discovery
from experiments import exp1_causal_flip
from experiments import exp2_conjunctive_rule
from experiments import exp3_planning
from experiments import exp4_curiosity

if __name__ == "__main__":
    exp0_entity_discovery.main()
    exp1_causal_flip.main()
    exp2_conjunctive_rule.main()
    exp3_planning.main()
    exp4_curiosity.main()
