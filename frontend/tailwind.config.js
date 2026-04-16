/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        /* ── Surface hierarchy ── */
        'surface':                '#121411',
        'surface-dim':            '#121411',
        'surface-bright':         '#383a36',
        'surface-container-lowest':'#0d0f0c',
        'surface-container-low':  '#1a1c19',
        'surface-container':      '#1e201d',
        'surface-container-high': '#292b27',
        'surface-container-highest':'#333532',
        'surface-variant':        '#333532',
        'surface-tint':           '#5bdda8',

        /* ── Primary ── */
        'primary':                '#5bdda8',
        'primary-container':      '#005c40',
        'primary-fixed':          '#7afac3',
        'primary-fixed-dim':      '#5bdda8',
        'on-primary':             '#003826',
        'on-primary-container':   '#57d9a5',
        'on-primary-fixed':       '#002114',
        'on-primary-fixed-variant':'#005138',
        'inverse-primary':        '#006c4b',

        /* ── Secondary ── */
        'secondary':              '#afd09a',
        'secondary-container':    '#324e24',
        'secondary-fixed':        '#cbedb4',
        'secondary-fixed-dim':    '#afd09a',
        'on-secondary':           '#1c3710',
        'on-secondary-container': '#9ebf8a',
        'on-secondary-fixed':     '#072100',
        'on-secondary-fixed-variant':'#324e24',

        /* ── Tertiary ── */
        'tertiary':               '#94d3c1',
        'tertiary-container':     '#175a4c',
        'tertiary-fixed':         '#afefdd',
        'tertiary-fixed-dim':     '#94d3c1',
        'on-tertiary':            '#00382e',
        'on-tertiary-container':  '#90cfbe',
        'on-tertiary-fixed':      '#00201a',
        'on-tertiary-fixed-variant':'#065043',

        /* ── Error ── */
        'error':                  '#ffb4ab',
        'error-container':        '#93000a',
        'on-error':               '#690005',
        'on-error-container':     '#ffdad6',

        /* ── Neutral / On-surface ── */
        'on-surface':             '#e3e3de',
        'on-surface-variant':     '#c2c9bb',
        'on-background':          '#e3e3de',
        'background':             '#121411',
        'outline':                '#8c9387',
        'outline-variant':        '#42493e',
        'inverse-surface':        '#e3e3de',
        'inverse-on-surface':     '#2f312e',

        /* ── Legacy compatibility ── */
        'green-ai': {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
      },
      fontFamily: {
        headline:  ['Space Grotesk', 'sans-serif'],
        body:      ['Plus Jakarta Sans', 'sans-serif'],
        label:     ['Plus Jakarta Sans', 'sans-serif'],
        technical: ['JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        DEFAULT: '0.5rem',
        lg:      '0.75rem',
        xl:      '1rem',
        '2xl':   '1.25rem',
        '3xl':   '1.5rem',
        full:    '9999px',
      },
    },
  },
  plugins: [],
};
