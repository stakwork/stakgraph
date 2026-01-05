import React from "react";

interface DashboardProps {
  title?: string;
}

export default function DashboardPage({ title = "Dashboard" }: DashboardProps) {
  return (
    <div className="dashboard">
      <h1>{title}</h1>
      <section className="stats">
        <div className="stat-card">
          <h3>Users</h3>
          <p>1,234</p>
        </div>
        <div className="stat-card">
          <h3>Revenue</h3>
          <p>$12,345</p>
        </div>
      </section>
    </div>
  );
}
