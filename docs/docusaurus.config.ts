import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const config: Config = {
  title: "KoopmanRL",
  tagline: "Koopman-Assisted Reinforcement Learning for Dynamical Systems",
  favicon: "img/favicon.svg",

  url: "https://dynamicslab.github.io",
  baseUrl: "/KoopmanRL/",

  organizationName: "dynamicslab",
  projectName: "KoopmanRL",

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          editUrl:
            "https://github.com/dynamicslab/KoopmanRL/tree/master/docs/",
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  stylesheets: [
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css",
      type: "text/css",
      integrity:
        "sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM",
      crossorigin: "anonymous",
    },
  ],

  themeConfig: {
    navbar: {
      title: "KoopmanRL",
      logo: {
        alt: "KoopmanRL Logo",
        src: "img/logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "startedSidebar",
          position: "left",
          label: "Get Started",
        },
        {
          type: "docSidebar",
          sidebarId: "documentationSidebar",
          position: "left",
          label: "Documentation",
        },
        {
          href: "https://github.com/dynamicslab/KoopmanRL",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            { label: "Quick Start", to: "/docs/quickstart" },
            { label: "Installation", to: "/docs/installation" },
            { label: "Algorithms", to: "/docs/algorithms/skvi" },
            { label: "Environments", to: "/docs/environments/linear-system" },
            { label: "API Reference", to: "/docs/api" },
          ],
        },
        {
          title: "Research",
          items: [
            {
              label: "Paper: KARL",
              href: "https://github.com/dynamicslab/KoopmanRL",
            },
            {
              label: "Zenodo Archive",
              href: "https://zenodo.org",
            },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/dynamicslab/KoopmanRL",
            },
            {
              label: "Issues",
              href: "https://github.com/dynamicslab/KoopmanRL/issues",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} KoopmanRL Contributors. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ["bash", "json", "python", "toml"],
    },
    colorMode: {
      defaultMode: "light",
      respectPrefersColorScheme: true,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
