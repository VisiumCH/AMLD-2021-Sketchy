import React from "react";
import { Switch, Route, Redirect } from "react-router-dom";
import Drawing from "./drawing.js";
import Embeddings from "./embeddingsPlot.js";

const App = () => {
  return (
    <Switch>
      <Redirect exact from="/" to="/drawing" />
      <Route path="/drawing">
        <Drawing />
      </Route>
      <Route path="/embeddings">
        <Embeddings />
      </Route>
    </Switch>
  );
};

export default App;
