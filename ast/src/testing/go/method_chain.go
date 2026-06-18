package main

// @ast node: Class "ServiceHandler"
// @ast edge: Operand -> Function "processItems" "method_chain.go"
// @ast node: DataModel "ServiceHandler"
// @ast node: Class "ItemStore"
// @ast edge: Operand -> Function "fetchItems" "method_chain.go"
// @ast node: DataModel "ItemStore"
// @ast node: Function "fetchItems"
// @ast node: Function "processItems"
// @ast edge: Calls -> Function "fetchItems" "method_chain.go"
// @ast node: Function "runPipeline"
// @ast edge: Calls -> Function "processItems" "method_chain.go"

type ServiceHandler struct {
	store *ItemStore
}

type ItemStore struct{}

func (s *ItemStore) fetchItems() []string {
	return []string{"item1", "item2"}
}

// processItems calls fetchItems via a 2-level selector: h.store.fetchItems()
func (h *ServiceHandler) processItems() []string {
	return h.store.fetchItems()
}

// runPipeline creates a ServiceHandler via short_var and calls processItems
func runPipeline() []string {
	h := &ServiceHandler{store: &ItemStore{}}
	return h.processItems()
}
