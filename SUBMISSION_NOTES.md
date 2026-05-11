# Submission Notes / What Was Fixed

## Major fixes

1. Removed the bundled `.venv` from the submission ZIP to keep the project clean and portable.
2. Replaced broken Tkinter/contextily dependency path with a Streamlit dashboard that matches the original README deployment style.
3. Added a correct `README.md` with running commands, Docker commands, algorithm mapping, and project structure.
4. Fixed path algorithms so they return structured results: path, cost, visited nodes, and weight used.
5. Added A* with a geographic heuristic instead of only duplicating Dijkstra behavior.
6. Added Time-dependent Dijkstra that uses `morning_travel_time`, `afternoon_travel_time`, `evening_travel_time`, or `night_travel_time`.
7. Rebuilt Kruskal MST using explicit union-find logic.
8. Improved road maintenance Dynamic Programming and generates repair cost/urgency from road condition, distance, and traffic.
9. Fixed public transit scheduling so `max_buses` is actually used as a real constraint.
10. Added Random Forest regression for traffic forecasting.
11. Added `run_demo.py` and `pytest` tests so the project can be verified quickly.

## How results differ

- Dijkstra: shortest distance route.
- A*: fastest travel-time route using heuristic guidance.
- Greedy: local cheapest-step route, not guaranteed optimal.
- Time-dependent Dijkstra: route changes according to selected hour/traffic period.
- MST: network design, not a start-to-end path.
- Maintenance DP: selected repair roads under budget.
- Transit DP: selected bus routes under available buses.
- Random Forest: predicts traffic count for a road and hour.


## Visualization Upgrade

- Replaced the plain longitude/latitude coordinate plot with a real OpenStreetMap GPS-style Cairo map.
- Added visible bus routes, bus-stop markers, metro lines, neighborhoods, facilities, selected algorithm routes, and MST overlay.
- Algorithms and data structures were kept unchanged; only the map visualization layer was upgraded.


## Final table/map fixes

- Added `road_name` beside `road_id` in Road Maintenance and MST output tables.
- Added human-readable stop names to the Public Transit selected-routes table.
- Fixed the Dokki/Cairo University coordinate overlap that made routes to Dokki appear to end at Cairo University.
- Route overlay now displays selected path node labels directly on the map.
