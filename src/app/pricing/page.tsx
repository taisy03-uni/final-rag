"use client";

import React from 'react';
import styles from './pricing.module.css';
import { MdCheck, MdArrowForward, MdAllInclusive, MdChat, MdSecurity, MdStars } from 'react-icons/md';

const PricingPage = () => {
  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Pricing Plans</h1>
      <p className={styles.subtitle}>Choose the plan that's right for you</p>
      
      <div className={styles.pricingGrid}>
        {/* Trial Plan */}
        <div className={styles.pricingCard}>
          <div className={styles.cardHeader}>
            <h2>Individual</h2>
            <div className={styles.price}>
              <span className={styles.currency}>£</span>
              <span className={styles.amount}>12</span>
              <span className={styles.decimal}>.99</span>
              <span className={styles.period}>/month</span>
            </div>
          </div>
          <div className={styles.cardContent}>
            <h3>Includes:</h3>
            <ul className={styles.benefitsList}>
              <li><MdCheck /> Basic AI Legal Assistant</li>
              <li><MdCheck /> 100 Queries per Month</li>
              <li><MdCheck /> Single Chat History</li>
              <li><MdCheck /> Standard Response Time</li>
              <li><MdCheck /> Email Support</li>
            </ul>
            <p className={styles.description}>
              Perfect for individuals wanting to experience the power of AI in legal research.
            </p>
            <button className={styles.ctaButton}>
              Go Individual <MdArrowForward />
            </button>
          </div>
        </div>

        {/* Premium Plan */}
        <div className={`${styles.pricingCard} ${styles.featured}`}>
          <div className={styles.cardHeader}>
            <h2>Premium</h2>
            <div className={styles.price}>
              <span className={styles.currency}>£</span>
              <span className={styles.amount}>31</span>
              <span className={styles.decimal}>.99</span>
              <span className={styles.period}>/month</span>
            </div>
          </div>
          <div className={styles.cardContent}>
            <h3>Everything in Basic, plus:</h3>
            <ul className={styles.benefitsList}>
              <li><MdAllInclusive /> Unlimited AI Queries</li>
              <li><MdChat /> Multiple Chat Sessions</li>
              <li><MdStars /> Priority Response Time</li>
              <li><MdCheck /> Advanced Legal Research Tools</li>
              <li><MdCheck /> 24/7 Priority Support</li>
            </ul>
            <p className={styles.description}>
              Ideal for professionals who need comprehensive legal AI assistance with no limits.
            </p>
            <button className={styles.ctaButton}>
              Go Premium <MdArrowForward />
            </button>
          </div>
        </div>

        {/* Enterprise Plan */}
        <div className={styles.pricingCard}>
          <div className={styles.cardHeader}>
            <h2>Enterprise</h2>
            <div className={styles.price}>
              <span className={styles.customPrice}>Custom Solution</span>
            </div>
          </div>
          <div className={styles.cardContent}>
            <h3>Custom Features:</h3>
            <ul className={styles.benefitsList}>
              <li><MdSecurity /> Private Deployment Options</li>
              <li><MdCheck /> Custom API Integration</li>
              <li><MdCheck /> Dedicated Account Manager</li>
              <li><MdCheck /> Custom Training & Onboarding</li>
              <li><MdCheck /> SLA & Premium Support</li>
            </ul>
            <p className={styles.description}>
              Tailored solutions for organizations needing custom deployment and integration.
            </p>
            <button className={styles.ctaButton}>
              Schedule Demo <MdArrowForward />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PricingPage;
