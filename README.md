# Reliability Assessment Simulation

This repo contains two Python scripts for power system reliability simulations based on the referenced article, using pandapower and related libraries.

## Scripts

- **simulate_9bus_risk_assessment.py**  
  Simulates a modified 9-bus system with single bus and breaker-and-a-half setups. Calculates fault risks and total risk, producing tables and figures as per the article.

- **simulate_68bus_risk_assessment.py**  
  Simulates a hypothetical 68-bus system focusing on tie-line faults. Outputs detailed risk tables without plotting. Parameter tuning may be needed.

## Notes

Both scripts use Dynamic Fault Tree simplifications and breaker-and-a-half adjustments with parameters from the article’s tables.

