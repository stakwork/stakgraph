interface AnalyticsEvent {
  name: string;
  properties?: Record<string, any>;
  timestamp: number;
}

interface UserIdentity {
  userId: string;
  traits?: Record<string, any>;
}

class AnalyticsService {
  private events: AnalyticsEvent[] = [];
  private currentUser: UserIdentity | null = null;

  track(event: string, properties?: Record<string, any>): void {
    const analyticsEvent: AnalyticsEvent = {
      name: event,
      properties,
      timestamp: Date.now(),
    };
    this.events.push(analyticsEvent);
    console.log("Analytics.track:", event, properties);
  }

  identify(userId: string, traits?: Record<string, any>): void {
    this.currentUser = { userId, traits };
    console.log("Analytics.identify:", userId, traits);
  }

  page(name: string, properties?: Record<string, any>): void {
    this.track(`page_view_${name}`, { page: name, ...properties });
    console.log("Analytics.page:", name, properties);
  }

  reset(): void {
    this.events = [];
    this.currentUser = null;
    console.log("Analytics.reset: cleared all data");
  }

  getEvents(): AnalyticsEvent[] {
    return [...this.events];
  }

  getCurrentUser(): UserIdentity | null {
    return this.currentUser;
  }

  flush(): void {
    console.log("Analytics.flush: sending", this.events.length, "events");
    this.events = [];
  }
}

export const analytics = new AnalyticsService();
