# Llm-Planner
Local LLM to find off-road routes

.\.venv\Scripts\activate

uvicorn app.main:app --port 9100

Swagger: http://localhost:9100/docs

Ollama: http://localhost:11434/

TASKKILL /PID 19240 /F

ollama pull qwen2.5:3b

--------------------------------------------------------------


Baseline Agent Behavior (Without LLM)
1. Route Collection

The Route-Agent periodically polls the blockchain and retrieves the most recent route of each vehicle.

2. Conflict Detection

Routes are converted into buffered corridors, and pairwise geometric intersections are computed.
If overlap exceeds a safety threshold, a conflict alert is generated.

3. Candidate Generation (Per Vehicle)

When a vehicle publishes a new route, the Agent requests multiple alternative routes for that same vehicle from its G-Nav instance:

POST /route/candidates


Typical candidates include:

A*

Dijkstra

Random (multiple runs)

Combined / Hybrid

Each candidate contains:

path: list of lat/lng points

metrics: e.g. length_m

Important:
Candidates are alternative routes for one specific vehicle, not different vehicles.

4. Candidate Ranking and Recommendation

Each candidate is evaluated against the routes of all other vehicles.

For every candidate, the Agent computes:

conflict_area_m2: total geometric overlap with other vehicles

length_m: route length

Scoring function:

score = conflict_area_m2 + 0.001 * length_m


The candidate with the lowest score becomes the Agent’s recommended route for that vehicle.

--------------------------------------------------------------

Role of the LLM

The Route-Agent can only choose among existing candidates.
In complex scenarios, all baseline algorithms may still produce suboptimal trade-offs (e.g., short routes with high conflict).

The LLM is introduced as a meta-planning component to address this limitation.

The LLM analyzes outcomes, not terrain:

Candidate algorithms

Route lengths

Conflict summaries

Based on this, it proposes how the cost function should be biased to potentially generate a better route.

LLM Output

Example LLM response:

{
  "best_candidate_algorithm": "dijkstra",
  "proposed_custom_algorithm": {
    "base_algorithm": "astar",
    "weights": {
      "w_slope": 0.8,
      "w_landcover": 1.0,
      "w_conflict": 3.0,
      "w_length": 0.001
    }
  }
}

Interpretation

best_candidate_algorithm
A non-binding opinion about the best existing candidate.

proposed_custom_algorithm
The core contribution:

Select a deterministic base algorithm (A* or Dijkstra).

Re-run it with a modified cost surface defined by weights.


Custom Route Computation (Cost Shaping)

Instead of using a fixed cost surface (e.g. COST_BASE = SLOPE + LANDCOVER_PENALTY), a new weighted cost matrix is constructed:

cost =
    w_slope     * SLOPE
  + w_landcover * LANDCOVER_PENALTY
  + w_conflict  * CONFLICT_MASK     (if provided)
  + w_danger    * DANGER_MASK       (if provided)


The base algorithm (e.g. A*) is then executed unchanged on this new cost matrix.

This produces a new candidate route for the same vehicle.


Full Agent + LLM Workflow

1. A vehicle publishes a new route.

2. The Route-Agent requests multiple baseline candidates from G-Nav.

3. The Agent ranks candidates using the standard scoring function.

4. The Agent sends candidate metrics and conflict summary to the LLM-Planner.

5. The LLM proposes a cost-shaping configuration (weights).

6. The Agent requests one additional route using these weights.

7. The new route is evaluated using the same scoring function.

8. If the new route is better, it is included; otherwise, it is discarded.

9. The final recommendation is always made by the Route-Agent.