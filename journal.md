- 19-03-2026
1. Verify claim: Curvature helps sheaf diffusion
- We have hints that Diagonal Sheaf Diffusion w/ d=1 doesn't lead to same oversmoothing behaviour of Bundle Sheaf Diffusion w/ d=4
- in Diagonal Sheaves we can allow for a curvature term, while we can't in a Bundle Sheaf
- we verify stability in curvature term of Diagonal Sheaf Diffusion, even w/ layers = 10
- question: does allowing flexibility for curvature help sheaf expressivity?
\\
2. New Point: in d=1, curvature on a graph happens to be highly negative in central graph's nodes -> hence Sheaf Laplacian induces hyperbolic geometry on the graph (!)
- in stalk d=1, we have better Diagonal Sheaf than w/ Bundles Sheaf (no curvature, by construction) -> allowing for the Laplacian to have a curvature term (which induces a new graph geometry) helps in the learning process (wrt/ over smoothing / over squashing)?
- preliminary results would suggest so, yet validation on multiple datasets is due
\\
3. Action points:
- more datasets (w/ bottlenecks) -> see [BUNDLE NEURAL NETWORKS FOR MESSAGE DIFFUSION ON GRAPHS](https://arxiv.org/pdf/2405.15540)
-