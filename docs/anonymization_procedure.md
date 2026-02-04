## Spatial Anonymization: Three-Stage Mental Model

This package follows a clear three-stage process for anonymizing household or cluster coordinates. Separating these stages avoids infinite loops, improves performance, and makes privacy guarantees explicit and testable.

---

### 1. Radius Selection (Privacy Envelope)

**Question answered:**  
*How large should the uncertainty region around the original point be?*

At this stage, the algorithm determines the **radius of anonymization** based on the chosen methodology.

Examples:
- **DHS (vanilla):** fixed maximum radius by urban/rural status
- **Donut geomasking:** fixed minimum and maximum radius
- **Population-adaptive:** radius increases until a population threshold is met
- **DHS + population:** population-adaptive radius capped by DHS maximum
- **Donut + population:** population-adaptive radius within donut bounds

Population logic belongs **only** at this stage.

> Population defines the *size* of the anonymization region, not the validity of a displaced point.

---

### 2. Point Displacement (Geometry)

**Question answered:**  
*Where inside the uncertainty region should the point be moved?*

Once the radius is determined, a displaced point is generated using geometric rules:
- random bearing
- DHS-style uniform distance
- donut (area-uniform annulus)
- geodesic displacement on the sphere

At this stage:
- population requirements are already satisfied
- no population checks are performed
- displacement is purely geometric

---

### 3. Constraint Evaluation (Validity)

**Question answered:**  
*Is the displaced point acceptable?*

After displacement, the candidate point is checked against **binary geometric constraints**, such as:
- inside administrative boundaries
- not in water bodies
- not on roads or major infrastructure (for household points)
- within country bounds

Constraints do **not** change the radius; they only accept or reject candidate points.  
If a constraint fails, a new displacement is attempted within the same radius.

---

### Key Design Principle

> **Population determines the anonymization envelope; constraints determine whether a displaced point is valid.**

Keeping these responsibilities separate:
- prevents infinite retry loops
- ensures predictable runtime
- allows clean composition of methods
- makes each stage independently testable
