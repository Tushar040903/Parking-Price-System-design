# 🚗 Dynamic Pricing for Urban Parking Lots

Capstone Project – Summer Analytics 2025  
Hosted by: **Consulting & Analytics Club × Pathway**

---

## 📌 Overview

This project simulates an intelligent real-time pricing system for urban parking lots based on fluctuating demand and environmental factors. We designed and implemented dynamic pricing models that adjust prices smoothly based on occupancy, queue length, traffic conditions, special events, vehicle type, and competition from nearby lots. The solution uses **streaming data** and is built using **only NumPy and Pandas** (no external ML libraries).

Our aim is to improve the utilization of limited urban parking spaces by avoiding underuse or congestion through data-driven pricing strategies.

---

## 🛠 Tech Stack

| Layer           | Technology                     |
|----------------|---------------------------------|
| Language        | Python                         |
| Data Handling   | Pandas, NumPy                  |
| Real-Time Engine| Pathway                        |
| Visualization   | Bokeh, Panel                   |
| Notebook IDE    | Google Colab                   |
| Diagram Tool    | Mermaid Live Editor            |

---

## 📊 Architecture Diagram

```mermaid
flowchart TD
    A[Historical CSV Dataset] -->|Replay as Stream| B[Pathway Stream Processor]
    B --> C[Preprocessing + Feature Extraction]
    C --> D[Model 1 - Linear Pricing]
    C --> E[Model 2 - Demand-Based Pricing]
    D --> F[Price Output Stream]
    E --> F
    F --> G["Bokeh Visualization - Panel App"]
