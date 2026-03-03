import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'WorldFlux',
  tagline: 'Unified Interface for World Models in Reinforcement Learning',
  favicon: 'img/logo.svg',

  url: 'https://worldflux.ai',
  baseUrl: '/',

  organizationName: 'worldflux',
  projectName: 'WorldFlux',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          path: '../docs',
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/worldflux/WorldFlux/tree/main/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'WorldFlux',
      logo: {
        alt: 'WorldFlux Logo',
        src: 'img/logo_transparent.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docs',
          position: 'left',
          label: 'Getting Started',
        },
        {
          to: '/api/factory',
          label: 'API Reference',
          position: 'left',
        },
        {
          to: '/reference/benchmarks',
          label: 'Reference',
          position: 'left',
        },
        {
          href: 'https://github.com/worldflux/WorldFlux',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Getting Started', to: '/getting-started/installation'},
            {label: 'API Reference', to: '/api/factory'},
            {label: 'Reference', to: '/reference/benchmarks'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub', href: 'https://github.com/worldflux/WorldFlux'},
            {label: 'Issues', href: 'https://github.com/worldflux/WorldFlux/issues'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} WorldFlux Contributors. Apache License 2.0.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml'],
    },
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
