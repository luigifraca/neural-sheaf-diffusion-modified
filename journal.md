### 🗓️ 19 March 2026

---

## 1. Claim Verification  
**Hypothesis:** *Curvature improves sheaf diffusion*

**Observations**
- Diagonal Sheaf Diffusion *(d = 1)* does **not** exhibit the same oversmoothing as Bundle Sheaf Diffusion *(d = 4)*  
- Diagonal sheaves **allow a curvature term**, whereas bundle sheaves do not  
- Stability of the curvature term holds even with **10 layers**

**Key Question**  
> Does curvature flexibility improve sheaf expressivity?

---

## 2. New Insight  
**Curvature induces geometry**

- For \( d = 1 \), curvature on graphs is **highly negative at central nodes**  
- ⇒ The Sheaf Laplacian induces a **hyperbolic geometry** on the graph

**Implications**
- Diagonal sheaves outperform bundle sheaves in \( d = 1 \)  
- Bundle sheaves lack curvature **by construction**  
- Introducing curvature modifies the graph geometry via the Laplacian  

**Working Hypothesis**
> Curvature helps mitigate **oversmoothing / oversquashing**

**Status**
- Preliminary results: ✅ promising  
- Validation: ❗ still needed across datasets  

---

## 3. Action Points

- Test on **more datasets with bottlenecks**  
  - Reference: [Bundle Neural Networks for Message Diffusion on Graphs](https://arxiv.org/pdf/2405.15540)  
- Discuss with **Alessio**  
  - Focus: continuous models + asymptotics  