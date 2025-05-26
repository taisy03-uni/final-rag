"use client";

import React from 'react';
import styles from './pricing.module.css';
import { MdCheck, MdArrowForward } from 'react-icons/md';

const PricingPage = () => {
  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Choose the plan that's right for you</h1>
      
      <div className={styles.pricingGrid}>
        {/* Solo Use Plan */}
        <div className={styles.pricingCard}>
          <div className={styles.cardHeader}>
            <h2>Solo Use</h2>
            <div className={styles.price}>
              <span className={styles.currency}>£</span>
              <span className={styles.amount}>10</span>
              <span className={styles.period}>/month</span>
            </div>
          </div>
          <div className={styles.cardContent}>
            <h3>Best For:</h3>
            <ul className={styles.benefitsList}>
              <li><MdCheck /> Legal Professionals</li>
              <li><MdCheck /> Independent Lawyers</li>
              <li><MdCheck /> Legal Consultants</li>
              <li><MdCheck /> Law Students</li>
              <li><MdCheck /> Paralegals</li>
            </ul>
            <p className={styles.description}>
              Perfect for individual legal professionals looking to enhance their practice with AI assistance.
            </p>
            <button className={styles.ctaButton}>
              Get Started <MdArrowForward />
            </button>
          </div>
        </div>

        {/* Enterprise Plan */}
        <div className={`${styles.pricingCard} ${styles.featured}`}>
          <div className={styles.cardHeader}>
            <h2>Enterprise</h2>
            <div className={styles.price}>
              <span className={styles.customPrice}>Custom Pricing</span>
            </div>
          </div>
          <div className={styles.cardContent}>
            <h3>Best For:</h3>
            <ul className={styles.benefitsList}>
              <li><MdCheck /> Law Firms</li>
              <li><MdCheck /> Legal Departments</li>
              <li><MdCheck /> Corporate Legal Teams</li>
              <li><MdCheck /> Private Practice Groups</li>
              <li><MdCheck /> Secure Private Deployment</li>
            </ul>
            <p className={styles.description}>
              Ideal for organizations seeking a private, secure AI system with dedicated support and customization options.
            </p>
            <button className={styles.ctaButton}>
              Contact Sales <MdArrowForward />
            </button>
          </div>
        </div>

        {/* Integrated System Plan */}
        <div className={styles.pricingCard}>
          <div className={styles.cardHeader}>
            <h2>Integrated System</h2>
            <div className={styles.price}>
              <span className={styles.customPrice}>Custom Pricing</span>
            </div>
          </div>
          <div className={styles.cardContent}>
            <h3>Best For:</h3>
            <ul className={styles.benefitsList}>
              <li><MdCheck /> Large Law Firms</li>
              <li><MdCheck /> Multi-National Corporations</li>
              <li><MdCheck /> Government Agencies</li>
              <li><MdCheck /> Custom API Integration</li>
              <li><MdCheck /> Private Data Integration</li>
            </ul>
            <p className={styles.description}>
              Complete customization and integration with your existing systems, including private data and custom workflows.
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
