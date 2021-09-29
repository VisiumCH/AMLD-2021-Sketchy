import React, { useState, useEffect } from "react";
import { Grid, GridItem, Text } from "@chakra-ui/react";
import Plot from "react-plotly.js";
import { heading } from "../ui_utils";
import { PageDrawer } from "../drawer.js";

function ScalarPerformance() {
  const [domainValues, setDomainValues] = useState([]);
  const [tripletValues, setTripletValues] = useState([]);
  const [lossValues, setLossValues] = useState([]);
  const [mapValues, setMapValues] = useState([]);
  const [map200Values, setMap200Values] = useState([]);
  const [precisionValues, setPrecisionValues] = useState([]);

  useEffect(() => {
    async function getValues() {
      // Send to back end
      const response = await fetch("/api/scalar_perf", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });

      if (response.ok) {
        const res = await response.json();
        setDomainValues(res["domain_loss"]);
        setTripletValues(res["triplet_loss"]);
        setLossValues(res["total_loss"]);
        setMapValues(res["map"]);
        setMap200Values(res["map_200"]);
        setPrecisionValues(res["prec_valid"]);
      }
    }
    getValues();
  }, []);

  function plotValues(title, values) {
    if (typeof values == "undefined") {
      values = [0];
    }
    return (
      <>
        <GridItem rowSpan={1} colSpan={1} align="center">
          <Plot
            data={[
              {
                type: "scatter",
                y: values,
                x: [...Array(values.length).keys()],
              },
            ]}
            layout={{
              width: 500,
              height: 350,
              title: title,
              hovermode: "closest",
              margin: {
                l: 60,
                r: 60,
                b: 60,
                t: 60,
              },
              font: {
                color: "darkBlue",
              },
              paper_bgcolor: "white",
              xaxis: { ticks: "outside", title: { text: "Epoch" } },
              yaxis: { ticks: "outside", title: { text: "Performance" } },
            }}
          />
        </GridItem>
      </>
    );
  }

  return (
    <>
      {heading}
      <Grid
        h="85.9vh"
        gap={4}
        align="center"
        templateRows="repeat(2, 1fr)"
        templateColumns="repeat(3, 1fr)"
      >
        {plotValues("Domain Loss", domainValues)}
        {plotValues("Triplet Loss", tripletValues)}
        {plotValues("Total Loss", lossValues)}
        {plotValues("MAP", mapValues)}
        {plotValues("MAP@200", map200Values)}
        {plotValues("Precision", precisionValues)}
      </Grid>
      {PageDrawer()}
      <Text fontSize="xs" color="darkBlue">
        I need a space.
      </Text>
    </>
  );
}

export default ScalarPerformance;
