# Spark Cloud Studio - Design Spec for Tunet Integration

## Brand Identity

Spark Cloud Studio uses a clean, modern SaaS aesthetic with purple as the primary accent color on a light/white background.

---

## Color Palette

### Primary
| Token | Hex | Usage |
|-------|-----|-------|
| `--spark-purple` | `#ae69f4` | Primary brand color, buttons, active states, links, accent borders |
| `--spark-purple-dark` | `#7E3AF2` | Hover states, darker emphasis |
| `--spark-purple-deep` | `#6C2BD9` | Pressed/active button states |
| `--spark-purple-gradient` | `linear-gradient(to right, #ae69f4, #c084fc)` | Gradient accents, progress bars |
| `--spark-purple-light` | `#F7F4FC` | Light purple tint backgrounds |

### Neutrals
| Token | Hex | Usage |
|-------|-----|-------|
| `--gray-900` | `#111827` | Primary text |
| `--gray-800` | `#1F2937` | Headings, strong text |
| `--gray-700` | `#374151` | Secondary text |
| `--gray-600` | `#4b5563` | Muted text |
| `--gray-500` | `#6b7280` | Placeholder text, icons |
| `--gray-400` | `#9ca3af` | Disabled text, subtle borders |
| `--gray-300` | `#D1D5DB` | Borders, dividers |
| `--gray-200` | `#e5e7eb` | Input borders, card borders |
| `--gray-100` | `#F9FAFB` | Background tint, table rows |
| `--white` | `#ffffff` | Page background, cards |

### Semantic
| Token | Hex | Usage |
|-------|-----|-------|
| `--blue` | `#1c64f2` | Info, secondary actions |
| `--green` | `#16A34A` | Success, "running" status |
| `--green-dark` | `#15803D` | Green hover |
| `--red` | `#EF4444` | Error, destructive, "stopped" |
| `--pink` | `#E74694` | Warning accent |
| `--navy` | `#020E49` | Dark backgrounds (hero) |

---

## Typography

### Font Stack
- **Primary/Body**: `Plus Jakarta Sans`, fallback to `ui-sans-serif, system-ui, sans-serif`
- **Monospace** (console, code, metrics): `Fira Mono`, fallback to `ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace`

### Scale
| Class | Size | Weight | Usage |
|-------|------|--------|-------|
| Page title | 24px (1.5rem) | 700 (bold) | Page headings ("Dashboard", "New Job") |
| Section title | 20px (1.25rem) | 600 (semibold) | Card/section headings ("Workstations", "File Storage") |
| Card title | 16px (1rem) | 600 | Card names ("Eric's 1st workstation") |
| Body | 14px (0.875rem) | 400 | Default text |
| Small | 12px (0.75rem) | 400 | Badges, labels, metadata |
| XS | 11px (0.6875rem) | 400 | Fine print, timestamps |

---

## Layout

### Sidebar Navigation
- **Width**: 200px fixed on desktop
- **Background**: White (`#ffffff`)
- **Border**: Right border `1px solid #e5e7eb`
- **Logo**: Spark logo top-left, full-color
- **Nav items**:
  - Icon (20px) + label, left-aligned
  - Padding: `12px 16px`
  - Font: 14px, `#374151` (gray-700)
  - **Active state**: `#ae69f4` text, light purple bg `#F7F4FC`, left border accent
  - **Hover**: Light gray bg `#F9FAFB`
- **Bottom section**: User avatar + name, Logout link

### Content Area
- **Background**: `#F9FAFB` (gray-100)
- **Max width**: Fluid, fills remaining space
- **Padding**: `24px` (desktop), `16px` (mobile)

### Cards
- **Background**: `#ffffff`
- **Border**: `1px solid #e5e7eb`
- **Border-radius**: `0.75rem` (12px) — rounded-xl
- **Shadow**: `0 1px 3px rgba(174, 105, 244, 0.1), 0 0 20px -5px rgba(174, 105, 244, 0.15)` (subtle purple glow)
- **Padding**: `20px 24px`
- **Hover** (interactive cards): `transform: scale(1.02)`, slightly elevated shadow

### Grid
- Dashboard uses responsive card grid
- Workstation cards: 3-column on large screens, stack on mobile
- Gap: `24px`

---

## Components

### Buttons

#### Primary (purple)
```css
background: #ae69f4;
color: #ffffff;
border: none;
border-radius: 0.5rem;
padding: 10px 20px;
font-weight: 600;
font-size: 14px;
cursor: pointer;
transition: background 0.2s;
```
- Hover: `#7E3AF2`
- Active: `#6C2BD9`
- Disabled: `opacity: 0.5; cursor: not-allowed`

#### Secondary (outline)
```css
background: transparent;
color: #ae69f4;
border: 1px solid #ae69f4;
border-radius: 0.5rem;
padding: 10px 20px;
font-weight: 600;
```
- Hover: `background: #F7F4FC`

#### Danger (red)
```css
background: #EF4444;
color: #ffffff;
border-radius: 0.5rem;
```

#### Ghost (text button)
```css
background: transparent;
color: #6b7280;
padding: 8px 12px;
```
- Hover: `background: #F9FAFB; color: #374151`

### Status Badges
- **Running/Active**: Green dot + "Running" text
  - Dot: `#16A34A`, 8px circle
  - Text: `#16A34A`, 12px, semibold
- **Queued**: Yellow/amber dot + "Queued"
- **Completed**: Purple dot + "Completed"
- **Stopped/Error**: Red dot + "Stopped"
- **Ready**: Red circle (like workstation cards) + "Ready to start"

Badge pill style:
```css
display: inline-flex;
align-items: center;
gap: 6px;
padding: 4px 10px;
border-radius: 9999px;
font-size: 12px;
font-weight: 500;
background: #F7F4FC;
```

### Form Inputs
```css
border: 1px solid #e5e7eb;
border-radius: 0.5rem;
padding: 10px 14px;
font-size: 14px;
font-family: Plus Jakarta Sans;
color: #111827;
background: #ffffff;
transition: border-color 0.2s;
```
- Focus: `border-color: #ae69f4; box-shadow: 0 0 0 3px rgba(174, 105, 244, 0.1)`
- Placeholder: `color: #9ca3af`

### Select/Dropdown
Same as input, with custom chevron icon on right.

### Checkbox (checked)
```css
background-color: #ae69f4;
```

### Progress Bar
```css
background: #e5e7eb; /* track */
border-radius: 9999px;
height: 8px;
```
Fill:
```css
background: linear-gradient(to right, #ae69f4, #c084fc);
border-radius: 9999px;
transition: width 0.3s;
```

### Drag-and-Drop Zone
```css
border: 2px dashed #D1D5DB;
border-radius: 0.75rem;
padding: 40px;
text-align: center;
background: #F9FAFB;
transition: all 0.3s ease;
```
Active/dragging:
```css
border-color: #ae69f4;
background: rgba(174, 105, 244, 0.05);
transform: scale(1.02);
```

### Metric Cards (File Storage, Cost)
- Large number: 28px, bold, `#111827`
- Label: 12px, `#6b7280`
- Sparkline/chart: Purple gradient fill

---

## Animations

### Slide-in (cards appearing)
```css
@keyframes slideIn {
  0% { opacity: 0; transform: translateY(-10px); }
  100% { opacity: 1; transform: translateY(0); }
}
animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
```

### Pulse (loading states)
```css
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}
animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
```

### Spinner
Purple (`#ae69f4`) rotating bars, 12-spoke pattern (as seen in loading screen)

### Card hover
```css
transition: transform 0.2s, box-shadow 0.2s;
transform: scale(1.02);
```

---

## Page Patterns

### Dashboard Pattern
- Top summary cards (metrics: storage usage, monthly cost) in a row
- Section title below
- Resource cards in a grid
- "Create new" card with dashed border and plus icon

### Form Page Pattern
- Page title at top
- Sections as cards stacked vertically
- Each section: GroupBox-like with section title, form fields below
- Sticky action bar at bottom with primary + secondary buttons

### Detail Page Pattern
- Page title + status badge in a row
- Tab bar below (if needed)
- Content area with mixed cards + data

---

## Responsive Breakpoints
- **sm**: 640px
- **md**: 768px (sidebar collapses)
- **lg**: 1024px (3-column grid)
- **xl**: 1280px

---

## Spark-Specific UI Elements

### Sidebar Nav Items (from screenshots)
1. Dashboard (home icon)
2. Billing & Usage
3. Manage Account
4. Organisation
5. ---separator---
6. File Storage
7. Workstation
8. Software
9. Render Farm
10. IntelliStory (external link)
11. ---bottom---
12. Help & Support
13. Logout
14. User avatar + name

### For Tunet Integration
Add under "Tools" section in sidebar:
- **Tunet Training** (brain/neural icon) — links to /tunet dashboard
