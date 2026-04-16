# Google Stitch Prompt - GreenAI Frontend Revamp

Create a full UI revamp for a web app called GreenAI, focused on sustainable machine learning and model compression for edge devices. Keep every existing feature and workflow, but redesign the visual language to feel modern, bold, and clearly Green AI themed.

## Design goals

1. Make it look like a sustainability-first AI operations product, not a generic dashboard.
2. Preserve all current functionality and information hierarchy.
3. Improve clarity of technical metrics like model size, accuracy, energy, and CO2 emissions.
4. Deliver responsive desktop and mobile layouts.

## Visual direction

1. Use a nature-tech aesthetic with layered gradients, subtle organic shapes, and clean data surfaces.
2. Primary palette: forest green, moss, mint, deep teal, warm neutrals. Avoid purple as a dominant color.
3. Typography: use expressive but readable fonts, for example Space Grotesk for headings, Plus Jakarta Sans for body text, and JetBrains Mono for numeric metrics.
4. Use meaningful motion only: page-load reveal, chart fade-ins, and progress transitions.
5. Keep accessibility high: WCAG-friendly contrast, clear focus states, keyboard-friendly controls.

## Global app shell requirements

1. Sticky top header with GreenAI brand mark, title, subtitle, FYP 2025-26 badge, and API Docs link.
2. Main content container with clear spacing rhythm and responsive breakpoints.
3. Footer with project context line including models, datasets, and compression techniques.

## Page 1: Dashboard (main page) must include

1. Loading state with centered spinner and backend reminder text.
2. Error state card for API connection failure with retry action.
3. Top stats cards:
- Baseline Accuracy
- Best Compression
- Smallest Model
- Device status (GPU or CPU) plus model/compression counts
4. Top-right actions:
- Compare Models on Images button
- Clear History button
5. Prepare Models panel:
- Ready count out of total
- Prepare All button
- Live progress bar and current model status
- Error state handling
6. Model Library grid:
- Cards for each baseline model
- Ready, not ready, and error states
- Model family visual identity
- Accuracy, size, training CO2 for ready models
- Badge showing saved compression result count per model
7. Expandable selected-model detail panel:
- Baseline inference details (accuracy, size, training CO2)
- Compression technique selector (Pruning, Quantization, Hybrid, Distillation)
- Compress selected button
- Compress by ALL button
- Running status text and error text
- Compression results panel with compressed-only metrics (size, accuracy, latency, CO2)
8. Dual analysis section:
- Compression Analysis chart card with tabbed views:
Size view, CO2 Emissions view, Radar view
- Strategy Comparison table with:
strategy name, accuracy, size, size reduction, train CO2, infer CO2, delete action
- Highlight baseline and best strategy rows
9. Energy and Carbon section:
- Baseline vs Compressed CO2 comparison rows
- Reduction percentage with color semantics
- Baseline CO2, compressed CO2, compressed train/infer CO2 fields
- Suspicious reduction warning state
- Baseline Energy Tracking cards including training energy savings
- Empty state when no data exists

## Page 2: Compare Models on Images must include

1. Header with title, subtitle, and Back to Dashboard action.
2. Unified error banner handling for initial load, form validation, and inference errors.
3. Step 1: Select Models
- Baseline model dropdown (prefer ready baselines)
- Compressed model dropdown
- Enable test-time augmentation toggle with explanatory help text
4. Step 2: Select Image Input
- Tabs: Sample Images and Upload Image
- Sample image grid selector with thumbnails and selected state
- Upload dropzone with drag-and-drop, choose file action, file type validation
5. Step 3: Preview and Run Inference
- Selected image preview panel with source label and source path or filename
- Compare Image button with loading and disabled states
6. Inference Results card:
- Baseline prediction card with top-3 classes and confidence
- Compressed prediction card with top-3 classes and confidence
- Prediction mismatch warning when classes differ
- Summary metrics: prediction match, confidence delta, input type
- Diagnostics and quality warnings section
7. Comparison History section:
- List of recent runs
- Clickable items to restore prior result view
- Clear History action

## Additional existing modules to preserve as optional screens/panels

1. Saved Models panel showing files, total size, compression badges.
2. Actions panel for Run Compression, Run Evaluation, Track Energy with status badges.
3. Legacy model upload compression wizard with model, method, dataset, epochs, and progress steps.

## Data visualization and metric formatting rules

1. Show tiny non-zero CO2 values as less than 0.000001 kg instead of 0.
2. Distinguish training CO2 vs inference CO2 vs baseline CO2 clearly.
3. Use chart legends and labels that are understandable to non-experts.
4. Keep tabular values right-aligned and monospaced for comparability.

## Interaction and persistence expectations

1. Design for auto-refresh dashboards and live-running statuses.
2. Support persistent local history for compression and image comparisons across refresh.
3. Include delete and clear actions with safe visual affordances.

## Technical implementation constraints for generated UI

1. Next.js + React + Tailwind CSS structure.
2. Recharts-friendly chart areas and card sizing.
3. Reusable components and tokens for scale.
4. Mobile-first responsive behavior with clean tablet and desktop expansion.

## Output requested from Stitch

1. A complete design system (colors, type scale, spacing, radii, shadows, states).
2. High-fidelity desktop and mobile screens for both main pages.
3. Component variants for all key cards, tables, selectors, and status states.
4. Motion guidance for loading, progress, and section transitions.
5. Final result should feel unmistakably Green AI: efficient, scientific, sustainable, and premium.