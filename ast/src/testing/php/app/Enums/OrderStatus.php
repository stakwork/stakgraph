<?php
// @ast node: Class "OrderStatus"
// @ast edge: Operand -> Function "label" "OrderStatus.php"
// @ast node: Function "label"

namespace App\Enums;

enum OrderStatus: string
{
    case Pending = 'pending';
    case Shipped = 'shipped';
    case Delivered = 'delivered';

    public function label(): string
    {
        return match ($this) {
            self::Pending => 'Pending',
            self::Shipped => 'Shipped',
            self::Delivered => 'Delivered',
        };
    }
}
