import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import HomepageFeatures from "@site/src/components/HomepageFeatures";
import Heading from "@theme/Heading";

import styles from "./index.module.css";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx("hero", styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className={styles.heroTitle}>
          {siteConfig.title}
        </Heading>
        <hr className={styles.heroRule} />
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <div>
          <Link className={styles.heroCta} to="/docs/quickstart">
            Get Started
          </Link>
          <Link
            className={styles.heroCta}
            to="https://github.com/dynamicslab/KoopmanRL"
          >
            GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Koopman-Assisted Reinforcement Learning — SKVI and SAKC algorithms for dynamical systems control"
    >
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
