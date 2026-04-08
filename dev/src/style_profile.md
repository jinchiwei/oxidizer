# Writing Style Profile: Jinchi Wei

## Corpus Analyzed
1. **M.S.E. Thesis** (Johns Hopkins, 2021): "Image Analysis Techniques for Scoliosis Using Deep Learning" (~50 pages)
2. **SPIE Conference Paper** (2024): "Intraoperative Tracked Ultrasound Imaging for Resolving Deformations During Spine Surgery" (~12 pages)
3. **Perspective/Review Paper** (UCSF, 2025): "Predicting ARIA Incidence Due to Anti-Amyloid Therapy for Alzheimer's Disease with Multimodal Risk Modeling Frameworks" (~8,000 words)

---

## 1. Statistical Analysis

### Sentence Length
- **Mean:** 24.1 words per sentence
- **Median:** 23 words
- **Standard deviation:** 10.1 (substantial natural variance)
- **Range:** 6 to 77 words
- **Distribution:**
  - 1–15 words: 17% (short, punchy sentences)
  - 16–25 words: 46% (core range)
  - 26–35 words: 27% (complex sentences)
  - 36–50 words: 8% (multi-clause constructions)
  - 51+ words: 2% (rare, long compound sentences)

The high standard deviation (10.1) means sentence length varies significantly — not a monotone rhythm. Short factual statements alternate with long, multi-clause constructions.

### Paragraph Length
- **Mean:** ~71 words (approximately 3–4 sentences per paragraph)
- **Range:** Very short (2-word headers/transitions) to 189 words
- Typical body paragraph: 4–8 sentences, often building from context to claim to evidence

### Vocabulary Complexity
- **Type-token ratio (ARIA paper):** 0.346 (high lexical diversity for technical writing)
- Heavy domain-specific vocabulary used precisely without over-explanation
- Abbreviations always defined on first use, then used consistently

### Voice
- **Active voice:** ~90%
- **Passive voice:** ~10%
- Active voice dominates, especially with "we" as subject
- Passive appears in methods sections describing experimental setups
- Examples of active: "We proposed," "We explored," "We generated," "We hypothesized"
- Examples of passive: "Data was categorized," "images were resized," "Vertebrae with simulated displacements were registered"

### Contractions
- **Zero** across all three documents (formal academic register throughout)

---

## 2. Pattern Extraction

### Transition Words/Phrases (Ranked by Frequency)
| Transition | Count | Context |
|---|---|---|
| "while" | 13 | Contrasting or qualifying within a sentence |
| "however" | 4 | Introducing counterpoints or complications |
| "notably" | 3 | Highlighting specific examples |
| "furthermore" | 2 | Adding supporting evidence |
| "as such" | 1+ | Drawing conclusions from prior statements |
| "additionally" | frequent | Adding parallel information |
| "similarly" | occasional | Drawing comparisons |
| "as a result" | occasional | Causal conclusions |
| "to address" | occasional | Introducing solutions |
| "on the contrary" | rare | Strong contrast |
| "lastly" | rare | Final items in sequence |
| "collectively" | rare | Summarizing groups of results |

**Pattern:** Transitions tend to be functional and straightforward. No decorative transitions. "While" is heavily favored over "although" or "whereas" for mid-sentence contrast.

### Paragraph Opening Patterns
Paragraphs typically open with one of these structures:
1. **Context-setting clause:** "As a disease, scoliosis is defined as..." / "For our tasks, a vertebrae can be considered..."
2. **Problem statement:** "Another important consideration for surgery is..." / "The negative results of these circumstances include..."
3. **Methodological framing:** "In selecting our data for segmentation, there was no easily accessible..." / "For our classification networks, we applied transfer learning..."
4. **Temporal/procedural:** "After creating the ground truth for our dataset..." / "When downloading the images..."
5. **Summary launch:** "Collectively, these experiments show..." / "The tasks that we explored showed..."

### Paragraph Closing Patterns
- Often ends with **implications or next steps**: "...this could make the task much more approachable"
- Sometimes closes with **specific results**: "...resulting in a median of 1.99 ± 0.46 mm error"
- Occasionally closes with **limitations acknowledged**: "...though the availability of standardized high-resolution MRI remains variable across contributing sites"

### Hedging vs. Directness
Mixed, but leans toward **qualified directness**:
- Makes claims but qualifies scope: "most likely," "would likely," "could theoretically"
- Uses "we hypothesized" before experimental claims
- Uses "we believed" for design decisions
- States results directly: "VGG-16 had the highest training, validation, and testing accuracies"
- Does NOT over-hedge — avoids "it is important to note that" or "it is worth mentioning"

### Punctuation Habits

**Semicolons:** Frequent. Used to join related independent clauses, especially when the second clause expands or qualifies the first.
> "observation without additional intervention if the Cobb angle is less than 20 degrees, bracing if 20-45 degrees, and surgery if greater than or equal to 45 degrees; the stages in between will reflect gradual steps toward ossification"

**Parenthetical asides:** Very heavy use — both for abbreviation definitions and for substantive qualifying information.
> "a collection of over 150,000,000 CT images without vertebral annotations, with the aim of creating our own synthetic X-rays as DRRs"
> "(including willingness to manually annotate individual vertebra)"

**Colons:** Used for lists and elaboration, not for dramatic effect.

**Em dashes (—):** **Never used in prose.** This is a distinctive absence.

**Commas:** Liberal. Used before "and" in series (inconsistent Oxford comma), and heavily in compound sentences with multiple clauses.

### Distinctive Word Choices and Repeated Phrases
- **"attained access to"** — used instead of "obtained" or "gained access to" (appears in both thesis and ARIA paper)
- **"holistic"** — used for comprehensive understanding ("holistic understanding of scoliosis severity")
- **"robust"** — used for model/system resilience (4 occurrences in ARIA paper alone)
- **"as well"** — frequently used at end of sentences as an additive
- **"to begin with"** — for initial conditions or baselines
- **"building upon"** / **"built upon"** — for extending prior work
- **"valuable and impactful"** — paired adjectives for significance
- **"scalable"** — for systems that generalize
- **"promising"** — for encouraging preliminary results
- **"readily"** — for things that are straightforwardly achievable
- **"we chose to focus on"** / **"we decided to"** — explaining design choices with rationale

### Structural Tendencies
- **Enumerated inline lists:** Frequently uses "1. ... 2. ... 3. ..." within sentence flow for methods steps
- **Citation-dense sentences:** Integrates references naturally within claims rather than stacking at end
- **Quantitative precision:** Always reports specific numbers, percentages, error margins (±), and sample sizes
- **Problem → solution → evidence** flow within sections
- **First person plural "we"** throughout — never "I" in the published work, "I" only in acknowledgments

---

## 3. Anti-AI Audit

### AI-isms Jinchi NATURALLY USES (legitimate in context)
These words appear in his writing but are used precisely and technically, not as filler:
| Word | Count | Notes |
|---|---|---|
| "robust" | 4 | Describing model performance — standard ML term |
| "leveraging" | 3 | Describing use of existing tools/data — standard |
| "paradigm" | 3 | "federated learning paradigm" — precise usage |
| "notably" | 3 | Highlighting specific examples with data |
| "scalable" | 2 | Technical systems description |
| "facilitate" | 2 | Describing what tools enable |
| "furthermore" | 2 | Standard transition |
| "comprehensive" | 1 | Describing datasets |
| "landscape" | 1 | "navigation landscape" — literal use |
| "seamlessly" | 1 | Describing clinical integration |

**Verdict:** These are fine to keep in the style guide. They appear at natural frequency and in precise contexts, not as filler or decoration.

### AI-isms Jinchi NEVER USES
| Word | Notes |
|---|---|
| "delve" / "delve into" | Never appears |
| "tapestry" | Never |
| "multifaceted" | Never |
| "pivotal" | Never |
| "groundbreaking" | Never |
| "cutting-edge" | Never |
| "transformative" | Never |
| "unravel" | Never |
| "plethora" | Never |
| "myriad" | Never |
| "foster" | Never |
| "harness" / "harnessing" | Never |
| "elevate" | Never |
| "streamline" | Never |
| "realm" | Never |
| "intricacies" | Never |
| "synergy" | Never |
| "innovative" | Never |
| "novel" | Never (despite ML papers commonly using this) |

### AI Detector Red Flags — Assessment

| Red Flag | Present in Jinchi's Writing? |
|---|---|
| Uniform sentence length | **NO** — SD of 10.1, wide range 6-77 words |
| Lack of sentence fragments | **Mostly yes** — writing is grammatically complete, but some informal structures in thesis acknowledgments |
| Overly smooth transitions | **NO** — transitions are functional, not decorative |
| No contractions | **YES** — but this is standard academic writing, not an AI tell |
| Excessive hedging | **NO** — hedges appropriately, states results directly |
| Em dash overuse | **NO** — never uses em dashes |
| "Furthermore/Moreover" stacking | **NO** — uses sparingly |
| Paragraph-opening "It is important to note" | **NEVER** |
| Lists of three with parallel structure | **Sometimes** — but with genuine content variation |
| Lack of specific numbers | **NO** — extremely number-dense |
| Generic concluding sentences | **Rare** — conclusions reference specific data or next steps |

**Summary:** Jinchi's writing naturally avoids most AI tells. The high sentence-length variance, quantitative density, domain specificity, and absence of decorative language make it resistant to AI detection. The only overlap is the absence of contractions, which is expected for academic writing.

---

## 4. Evolution Across Documents (2021–2025)

### Thesis (2021)
- More tentative, qualifying statements ("we believed," "we assumed")
- Longer explanatory passages for basic concepts
- More detailed step-by-step methodology descriptions
- Student voice: explaining design decisions with reasoning

### SPIE Paper (2024)
- Most concise of the three
- Confident, direct assertions
- Dense technical writing with minimal hedging
- Professional conference tone

### ARIA Paper (2025)
- Most sophisticated vocabulary and sentence construction
- More comfortable with longer, multi-clause sentences
- Integrates broader clinical and policy context
- Uses slightly more "elevated" vocabulary (e.g., "ubiquitous," "imperative," "plausibly")
- Review/perspective register vs. experimental report

**Trend:** Writing has become more confident, more complex in sentence structure, and broader in scope over time while maintaining the same core patterns (no em dashes, heavy parentheticals, "we" voice, quantitative precision).

---

## 5. Verbatim Excerpts (Representative Samples)

**Excerpt 1 — Problem framing with quantitative stakes (Thesis, Introduction):**
> "The negative results of these circumstances include inefficient delegation of physician efforts and time, billions of dollars in unnecessary clinical spending, and ultimately poorer health outcomes. Any tools that can encourage consistency and precision in making these decisions would alleviate the downsides of current referral pipelines."

**Excerpt 2 — Methodological reasoning (Thesis, Ch. 2):**
> "We realized there were two main problems with our goal of both segmenting and anatomically labeling individual vertebra: 1. Vertebra are not very distinct in appearance from their neighbors, creating problems distinguishing vertebra from each other, and 2. ground truth in 3D generally includes all of the vertebrae, including body and projections, as shown in Figure 1.2."

**Excerpt 3 — Results with precision (SPIE Paper):**
> "Registration error using only the transverse scans was 1.43 ± 0.30 mm. In this scenario, our longitudinal scans were significantly shorter in length, leading to a higher registration error of 16.62 ± 7.04 mm. Though they were sparse, when combined with transverse information, performance increased, with a final error of 1.22 ± 0.39 mm."

**Excerpt 4 — Review-style synthesis (ARIA Paper):**
> "Compromised vessel integrity can result in vasogenic edema and extraparenchymal effusion (ARIA-E), microhemorrhages and superficial siderosis (ARIA-H), or a combination thereof. Both ARIA-E and ARIA-H can occur concomitantly or independently, and while often asymptomatic and reversible in the case of ARIA-E, can lead to severe complications, including seizures, strokes, and death."

**Excerpt 5 — Discussion with honest limitations (Thesis, Ch. 3):**
> "For our Stage 4 data during testing, for example, while only 2 images were correctly classified, 11/16 images were classified as Stages 3 or 5, showing that the network recognized the true labels were around 4 but was not precise enough to make the final distinction. With more high-quality training cases, this would most likely be remedied."
