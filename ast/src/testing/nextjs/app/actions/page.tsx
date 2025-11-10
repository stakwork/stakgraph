"use client";
import { useActions } from "../../lib/hooks/useActions";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
} from "../../components/ui/card";
import { Button } from "../../components/ui/button";

function ActionsDemo() {
  const actionsObj = useActions();
  const { addForce, removeForce, clearAll, getCount } = useActions();

  const handleAddViaObject = () => {
    actionsObj.addForce("cluster");
  };

  const handleRemoveViaObject = (id: string) => {
    actionsObj.removeForce(id);
  };

  const handleClearViaObject = () => {
    actionsObj.clearAll();
  };

  const handleCountViaObject = () => {
    const count = actionsObj.getCount();
    console.log("Count via object:", count);
  };

  const handleAddViaDestructured = () => {
    addForce("gravity");
  };

  const handleRemoveViaDestructured = (id: string) => {
    removeForce(id);
  };

  const handleClearViaDestructured = () => {
    clearAll();
  };

  const handleCountViaDestructured = () => {
    const count = getCount();
    console.log("Count via destructured:", count);
  };

  return (
    <main className="max-w-4xl mx-auto py-8">
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Object Access Pattern</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                Using: <code>actionsObj.method()</code>
              </p>

              <div className="space-y-2">
                <Button onClick={handleAddViaObject} className="w-full">
                  Add Force (Object)
                </Button>
                <Button onClick={handleClearViaObject} className="w-full">
                  Clear All (Object)
                </Button>
                <Button onClick={handleCountViaObject} className="w-full">
                  Get Count (Object)
                </Button>
              </div>

              <div>
                <h4 className="font-medium mb-2">Actions (Object Pattern)</h4>
                {actionsObj.actions.length === 0 ? (
                  <p className="text-sm text-gray-500">No actions</p>
                ) : (
                  <ul className="space-y-1">
                    {actionsObj.actions.map((action) => (
                      <li
                        key={action.id}
                        className="flex justify-between items-center text-sm"
                      >
                        <span>{action.type}</span>
                        <Button
                          onClick={() => handleRemoveViaObject(action.id)}
                          variant="destructive"
                          size="sm"
                        >
                          Remove
                        </Button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Destructured Pattern</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                Using: <code>method()</code> directly
              </p>

              <div className="space-y-2">
                <Button
                  onClick={handleAddViaDestructured}
                  className="w-full"
                  variant="outline"
                >
                  Add Force (Destructured)
                </Button>
                <Button
                  onClick={handleClearViaDestructured}
                  className="w-full"
                  variant="outline"
                >
                  Clear All (Destructured)
                </Button>
                <Button
                  onClick={handleCountViaDestructured}
                  className="w-full"
                  variant="outline"
                >
                  Get Count (Destructured)
                </Button>
              </div>

              <div>
                <h4 className="font-medium mb-2">
                  Note: Both patterns affect the same hook instance
                </h4>
                <p className="text-xs text-gray-500">
                  The destructured pattern typically works because the functions
                  are directly imported/accessible. The object pattern is what
                  we're testing for proper call resolution.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}

export { ActionsDemo as default };
