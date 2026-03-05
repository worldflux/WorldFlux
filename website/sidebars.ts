import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    'index',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/quickstart',
        'getting-started/cpu-success',
        'getting-started/concepts',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/factory',
        'api/protocol',
        'api/state',
        'api/training',
        'api/config-reference',
        'api/factory-reference',
        'api/protocol-reference',
        'api/training-reference',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/model-evaluation',
        'guides/training-observability',
      ],
    },
    // Tutorials hidden until content is production-ready (see tutorial-policy.md)
    // {
    //   type: 'category',
    //   label: 'Tutorials',
    //   items: [
    //     'tutorials/train-first-model',
    //     'tutorials/reproduce-dreamer-tdmpc2',
    //     'tutorials/dreamer-vs-tdmpc2',
    //   ],
    // },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/benchmarks',
        'reference/config',
        'reference/observation-action',
        'reference/docs-stack',
        'reference/extensibility',
        'reference/quality-gates',
        'reference/release-checklist',
        'reference/parity',
        'reference/publishing',
        'reference/troubleshooting',
        'reference/tutorial-policy',
        'reference/unified-comparison',
        'reference/wasr',
      ],
    },
  ],
};

export default sidebars;
