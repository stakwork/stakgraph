class ProcessPaymentJob < ApplicationJob
  queue_as :default

  def perform(order_id, amount)
    # Payment processing logic here
    order = Order.find(order_id)
    PaymentService.charge(order, amount)
  end
end
