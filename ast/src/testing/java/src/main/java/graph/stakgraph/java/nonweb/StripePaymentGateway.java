package graph.stakgraph.java.nonweb;

import java.util.HashMap;
import java.util.Map;

public class StripePaymentGateway implements PaymentGateway {
    private final Map<String, Long> accountBalances = new HashMap<>();

    public StripePaymentGateway() {
        accountBalances.put("seed", 10_000L);
    }

    @Override
    public boolean charge(String accountId, long cents) {
        Long current = accountBalances.getOrDefault(accountId, 0L);
        if (current < cents) {
            return false;
        }
        accountBalances.put(accountId, current - cents);
        return true;
    }
}
