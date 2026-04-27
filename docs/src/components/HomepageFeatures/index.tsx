import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: "Koopman-Assisted RL",
    description: (
      <>
        Two novel KARL algorithms — Soft Koopman Value Iteration (SKVI) and
        Soft Koopman Actor-Critic (SAKC) — leverage the Koopman operator to
        linearize nonlinear dynamics for improved sample efficiency.
      </>
    ),
  },
  {
    title: "Control-Oriented Environments",
    description: (
      <>
        Four benchmark environments rooted in the control literature: Linear
        System, Fluid Flow, Lorenz attractor, and Double-Well potential. Each
        environment targets a distinct class of dynamical behavior.
      </>
    ),
  },
  {
    title: "Modular & Reproducible",
    description: (
      <>
        Built on CleanRL-style implementations with typed argument parsers,
        Optuna-based hyperparameter search, and pre-optimized configurations
        — making every experiment reproducible and easy to extend.
      </>
    ),
  },
];

function Feature({ title, description }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
