# Must answer for the assignment
- [ ] Implement the Kármán vortex street (FD, LBM, FEM)
- [ ] How does computational cost compare across the three methods?
- [ ] How do accuracy and numerical stability scale with mesh resolution?
- [ ] How does accuracy differ near curved objects?
- [ ] How easy is each method to implement?
- [ ] Is the implementation suitable for parallel execution?
- [ ] Would the implementation fit GPU execution?

# LBM - I want to figure out
- [ ] Does Re = 150 reproduce the classic Kármán vortex street?
- [ ] Do different U_inlet values at the same Re yield the same result?
- [ ] What is BGK? How does it differ from LBGK?
- [ ] What is the impact of a small transverse perturbation? What is the maximum allowable value? What happens when it varies, including zero?
- [ ] What is the data structure of f? Is it a good design?
- [ ] Step 7e is still unclear: is it really the correct implementation approach?
- [ ] Step 7f is still unclear: is it really the correct implementation approach?
- [ ] How much finer chaotic detail appears at high Re?

For LBM, focus on `lbm_karman-Drag_lift.py` from all LBM scripts provided by Gabor is sufficient.

### Key Comparisons to Make
| Aspect | FD | LBM | FEM |
|--------|-----|-----|-----|
| Accuracy near curved surfaces | ?? | Bounce-back artifacts | ?? |
| Implementation complexity | Moderate?? | Simple | Complex (but ngsolve helps) |
| Parallelization | Straightforward | Excellent (local operations) | ?? or non-trivial |
| Max stable Re | Limited by CFL | Limited by Ma << 1 (??) | Highest (implicit) (??) |
| Computational cost | Low per step | Low per step | High per step, fewer steps?? |
| GPU suitability | Good/OK | Excellent | Difficult?? |