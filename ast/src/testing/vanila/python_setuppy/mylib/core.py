from typing import Dict, Any
from .models import Item, ProcessResult
from .utils import log_process

class Processor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @log_process
    def process(self, data: Dict[str, Any]) -> ProcessResult:
        """
        Validates and processes the input data.
        """
        if not data.get("items"):
            return ProcessResult(success=False, processed_count=0, errors=["No items found"])
        
        count = 0
        errors = []
        
        for item_data in data["items"]:
            try:
                item = Item(**item_data)
                # Simulate processing logic
                print(f"Processing {item.name}")
                count += 1
            except Exception as e:
                errors.append(str(e))
                
        return ProcessResult(
            success=len(errors) == 0,
            processed_count=count,
            errors=errors
        )

def main():
    """CLI entry point"""
    processor = Processor()
    # Mock data
    data = {
        "items": [
            {"id": 1, "name": "Item 1", "tags": ["a", "b"]},
            {"id": 2, "name": "Item 2"}
        ]
    }
    result = processor.process(data)
    print(f"Processed: {result.processed_count}, Success: {result.success}")

if __name__ == "__main__":
    main()
