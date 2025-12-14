// @ts-nocheck
import { analytics } from "../../lib/services/analytics";

describe("unit: analytics service", () => {
  beforeEach(() => {
    analytics.reset();
  });

  it("tracks events", () => {
    analytics.track("button_click", { button: "submit" });

    const events = analytics.getEvents();
    expect(events.length).toBe(1);
    expect(events[0].name).toBe("button_click");
    expect(events[0].properties?.button).toBe("submit");
    console.log("analytics.track called:", events[0]);
  });

  it("identifies user", () => {
    analytics.identify("user-123", { name: "John Doe", plan: "premium" });

    const user = analytics.getCurrentUser();
    expect(user?.userId).toBe("user-123");
    expect(user?.traits?.name).toBe("John Doe");
    console.log("analytics.identify called:", user);
  });

  it("tracks page views", () => {
    analytics.page("Home", { referrer: "google" });

    const events = analytics.getEvents();
    expect(events.length).toBe(1);
    expect(events[0].name).toBe("page_view_Home");
    console.log("analytics.page called:", events[0]);
  });

  it("resets analytics data", () => {
    analytics.track("event1");
    analytics.track("event2");
    analytics.identify("user-123");

    analytics.reset();

    const events = analytics.getEvents();
    const user = analytics.getCurrentUser();
    expect(events.length).toBe(0);
    expect(user).toBeNull();
    console.log("analytics.reset called: data cleared");
  });

  it("flushes events", () => {
    analytics.track("event1");
    analytics.track("event2");
    analytics.track("event3");

    analytics.flush();

    const events = analytics.getEvents();
    expect(events.length).toBe(0);
    console.log("analytics.flush called: events sent");
  });
});

describe("unit: analytics workflows", () => {
  beforeEach(() => {
    analytics.reset();
  });

  it("handles user journey tracking", () => {
    analytics.identify("user-456", { name: "Jane Smith" });
    analytics.page("Home");
    analytics.track("click_signup");
    analytics.page("Signup");
    analytics.track("submit_form", { form: "signup" });

    const events = analytics.getEvents();
    expect(events.length).toBe(4);

    const user = analytics.getCurrentUser();
    expect(user?.userId).toBe("user-456");

    console.log("User journey tracked:", events.length, "events");
  });

  it("chains multiple tracking calls", () => {
    analytics.track("view_product", { productId: "p-1" });
    analytics.track("add_to_cart", { productId: "p-1" });
    analytics.track("view_cart");
    analytics.track("checkout_start");

    const events = analytics.getEvents();
    expect(events.length).toBe(4);
    console.log("Chained tracking calls:", events.length);
  });

  it("handles identify and track together", () => {
    analytics.identify("user-789", { email: "user@example.com" });
    analytics.track("login_success");
    analytics.page("Dashboard");

    const events = analytics.getEvents();
    const user = analytics.getCurrentUser();

    expect(events.length).toBe(2);
    expect(user?.userId).toBe("user-789");

    console.log("Identify + track workflow completed");
  });
});

describe("unit: analytics batching", () => {
  beforeEach(() => {
    analytics.reset();
  });

  it("accumulates events before flush", () => {
    analytics.track("event1");
    analytics.track("event2");
    analytics.track("event3");

    let events = analytics.getEvents();
    expect(events.length).toBe(3);

    analytics.flush();

    events = analytics.getEvents();
    expect(events.length).toBe(0);

    console.log("Events accumulated and flushed");
  });

  it("tracks after flush", () => {
    analytics.track("before_flush");
    analytics.flush();

    analytics.track("after_flush");

    const events = analytics.getEvents();
    expect(events.length).toBe(1);
    expect(events[0].name).toBe("after_flush");

    console.log("Tracking continues after flush");
  });
});

describe("unit: analytics properties", () => {
  beforeEach(() => {
    analytics.reset();
  });

  it("tracks events with complex properties", () => {
    analytics.track("purchase", {
      items: ["item1", "item2"],
      total: 99.99,
      currency: "USD",
      metadata: { coupon: "SAVE10" },
    });

    const events = analytics.getEvents();
    expect(events[0].properties?.total).toBe(99.99);
    expect(events[0].properties?.items).toEqual(["item1", "item2"]);

    console.log("Complex properties tracked:", events[0].properties);
  });

  it("tracks page with properties", () => {
    analytics.page("Product", {
      productId: "p-123",
      category: "electronics",
      price: 299.99,
    });

    const events = analytics.getEvents();
    expect(events[0].properties?.productId).toBe("p-123");

    console.log("Page with properties tracked");
  });
});
