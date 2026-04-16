# Claude Prompt: Apply Google Stitch UI to Existing GreenAI Frontend (UI-Only Refactor)

You are working in this repository and must **revamp the entire frontend UI** using the provided Stitch design assets while keeping all current API integrations and behavior exactly the same.

## Primary Goal

Transform the visual design of the current Next.js frontend to match the new Stitch design language from:

1. `Dashboard Screen/DESIGN.md`
2. `Dashboard Screen/code.html`
3. `Dashboard Screen/screen.png`
4. `Model Comparison Screen/DESIGN.md`
5. `Model Comparison Screen/code.html`
6. `Model Comparison Screen/screen.png`

This is a **visual/UI refactor only**. Keep all logic, API calls, payloads, feature behavior, state flows, and persistence semantics unchanged.

---

## Non-Negotiable Constraints

1. Do **not** remove, rename, or change any API endpoint usage.
2. Do **not** change request/response contracts.
3. Do **not** change business logic or data transformations.
4. Do **not** change localStorage keys or persistence behavior.
5. Do **not** break polling, loading, error, empty, or disabled states.
6. Keep every current user flow working exactly as before.
7. Keep CO2/energy formatting semantics exactly intact (including tiny non-zero handling).
8. Keep comparison and compression history behavior intact.

If a design requirement conflicts with existing functionality, preserve functionality first and adapt visuals around it.

---

## Functional Invariants You Must Preserve

### Dashboard flow (`/`)

1. Initial loading state with spinner and API reminder.
2. API error state with retry action.
3. Stats cards behavior and values.
4. Prepare panel behavior, progress polling, and error handling.
5. Model library grid selection behavior.
6. Compression dialog behavior:
- Technique selection
- Compress selected
- Compress by ALL
- Running status and errors
- Result fetch from history/status
7. Accumulated compression history behavior.
8. Clear history behavior.
9. Chart tabs and rendering behavior.
10. Comparison table behavior including delete action.
11. Energy/CO2 section behavior and warning logic.

### Model comparison flow (`/model-comparison`)

1. Initial loading/options/samples fetch behavior.
2. Baseline/compressed model selection behavior.
3. Sample vs upload tab behavior.
4. Upload validation and preview behavior.
5. TTA toggle behavior.
6. Compare action behavior and payload semantics.
7. Result rendering, mismatch warning, diagnostics warnings.
8. History persistence, restore, and clear behavior.

### Persistence and formatting invariants

1. Keep localStorage keys unchanged:
- `compressionHistory`
- `greenai_compression_results`
- `greenai_compare_image_last_result`
- `greenai_compare_image_history`
2. Keep tiny CO2 formatting behavior (`<0.000001 kg` for tiny non-zero values).
3. Keep baseline/compressed emissions semantics intact.

---

## Files to Update (UI/Styling)

Apply the redesign across these existing frontend files:

1. `frontend/src/app/globals.css`
2. `frontend/src/app/layout.tsx`
3. `frontend/src/app/page.tsx`
4. `frontend/src/app/model-comparison/page.tsx`
5. `frontend/src/components/StatsCards.tsx`
6. `frontend/src/components/PreparePanel.tsx`
7. `frontend/src/components/ModelGrid.tsx`
8. `frontend/src/components/CompressionDialog.tsx`
9. `frontend/src/components/CompressionResults.tsx`
10. `frontend/src/components/ComparisonTable.tsx`
11. `frontend/src/components/CompressionChart.tsx`
12. `frontend/src/components/EnergySection.tsx`
13. `frontend/src/components/model-comparison/ImageSourceTabs.tsx`
14. `frontend/src/components/model-comparison/SampleImageSelector.tsx`
15. `frontend/src/components/model-comparison/ImageUploadDropzone.tsx`
16. `frontend/src/components/model-comparison/SelectedImagePreview.tsx`
17. `frontend/src/components/model-comparison/PredictionComparisonResults.tsx`
18. `frontend/tailwind.config.js` (only for theme/token support)

You may also style these legacy components to stay visually consistent:

1. `frontend/src/components/ActionPanel.tsx`
2. `frontend/src/components/DynamicResults.tsx`
3. `frontend/src/components/ModelDashboard.tsx`
4. `frontend/src/components/ModelsPanel.tsx`
5. `frontend/src/components/ModelUpload.tsx`

---

## Files You Must Not Change (Logic/API Layer)

Do not edit these unless absolutely unavoidable for purely presentational wiring (no logic changes allowed):

1. `frontend/src/lib/api.ts`
2. `frontend/src/lib/hooks/useCompareImage.ts`
3. `frontend/src/middleware.ts`

If you touch any of these, explain exactly why and prove no behavior changed.

---

## Target Design System to Implement

Follow the Stitch design language from both `DESIGN.md` files and both `code.html` files:

1. Dark nature-tech palette (forest, mint, teal, warm dark neutrals).
2. Typography hierarchy:
- Space Grotesk for display/headlines
- Plus Jakarta Sans for body
- JetBrains Mono for technical values
3. Tonal layering over hard borders.
4. Glassmorphism for floating elements where appropriate.
5. Gradient CTA treatment for primary actions.
6. Organic but disciplined rounded shapes.
7. Strong metric readability and contrast.
8. Responsive behavior for desktop/tablet/mobile.

Avoid generic template styling. Match the provided screen look and mood.

---

## Implementation Rules

1. Keep all component props and event handlers intact.
2. Prefer changing class names, wrappers, spacing, and visual structure.
3. Do not alter API call timing, intervals, or state machine behavior.
4. Preserve current conditional rendering logic.
5. Preserve button disabled conditions and loading labels.
6. Preserve all user-visible functional text that conveys operational state.
7. Keep chart data wiring exactly the same; only restyle chart container, colors, and visual presentation.
8. Keep accessibility intact (focus states, keyboard usage, readable contrast).

---

## Required Validation Before Finishing

After implementation, run and confirm:

1. `cd frontend && npm run build` passes.
2. No TypeScript errors introduced.
3. Dashboard still supports:
- model selection
- compression execution
- history save/delete/clear
- chart/tab switching
- energy section rendering
4. Model comparison still supports:
- sample select
- upload flow
- compare action
- history restore/clear
- warnings display

---

## Output Format You Should Return

When done, provide:

1. List of files changed.
2. Short summary of visual changes per major section.
3. Explicit confirmation that API/logic was not changed.
4. Build/test result.
5. Any known UI-only limitations.

---

## Important Reminder

This task is **not** a frontend rewrite and **not** a feature rewrite.

It is a **full visual redesign** mapped onto the existing production behavior.

Keep functionality identical; change only the presentation layer.