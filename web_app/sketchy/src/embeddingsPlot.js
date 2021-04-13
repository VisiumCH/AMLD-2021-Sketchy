import React, { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import Plot from "react-plotly.js";
import {
  Box,
  ChakraProvider,
  Button,
  Text,
  Heading,
  VStack,
  Grid,
  GridItem,
} from "@chakra-ui/react";
import {
  gray,
  darkGray,
  textColor,
  backgroundColor,
  buttonHeight,
  buttonWidth,
  colors,
} from "./constants";
import { BiPencil, BiImages } from "react-icons/bi";

function Embeddings() {
  const { state } = useLocation();
  const [result, setResult] = useState({});
  const [nbDimensions, setNbDimensions] = useState(3);
  let traces = [];

  useEffect(() => {
    let to_send = { nb_dim: nbDimensions };
    if (typeof state !== undefined) {
      to_send["sketch"] = state;
    }

    async function getEmbeddings(to_send) {
      // Send to back end
      const response = await fetch("/get_embeddings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(to_send),
      });

      if (response.ok) {
        const res = await response.json();
        setResult(res);
      }
    }

    getEmbeddings(to_send);
  }, [state, nbDimensions]);

  function fillTraces() {
    let marker_size = 4;
    let i = 0;
    for (let key in result) {
      if (key === "My Custom Sketch") {
        marker_size = 10;
      } else {
        marker_size = 6;
      }
      let trace = {};
      if (nbDimensions === 3) {
        trace = {
          x: result[key]["x"],
          y: result[key]["y"],
          z: result[key]["z"],
          name: key,
          type: "scatter3d",
          mode: "markers",
          marker: {
            color: colors[i],
            size: marker_size,
          },
          hoverinfo: "name",
        };
      } else {
        trace = {
          x: result[key]["x"],
          y: result[key]["y"],
          name: key,
          type: "scatter",
          mode: "markers",
          marker: {
            color: colors[i],
            size: marker_size * 2,
          },
          hoverinfo: "name",
        };
      }
      traces.push(trace);
      i = i + 1;
    }
  }
  fillTraces();

  function getDimensionButton() {
    let text = "Switch to 3D";
    let dim = 3;
    if (nbDimensions === 3) {
      text = "Switch to 2D";
      dim = 2;
    }
    return (
      <Button
        color={backgroundColor}
        border="2px"
        borderColor={darkGray}
        variant="solid"
        size="lg"
        height={buttonHeight}
        width={buttonWidth}
        onClick={() => {
          setNbDimensions(dim);
        }}
      >
        {text}
      </Button>
    );
  }

  return (
    <ChakraProvider>
      <Box bg={backgroundColor}>
        <VStack spacing={4} align="center">
          <Heading fontSize="4xl" color={textColor} align="center">
            AMLD 2021 Visium's Sketchy App
          </Heading>
          <Text fontSize="2xl" color={textColor} align="center">
            Embeddings: Images and Sketches in latent space
          </Text>
        </VStack>

        <Grid
          h="90vh"
          w="98vw"
          gap={4}
          align="center"
          templateRows="repeat(3, 1fr)"
          templateColumns="repeat(7, 1fr)"
        >
          <GridItem rowSpan={1} colSpan={1}>
            <VStack spacing={3} direction="row" align="center">
              <Text fontSize="2xl" color={textColor} align="center">
                Dimension: {nbDimensions}D
              </Text>
              {getDimensionButton()}
            </VStack>
          </GridItem>
          <GridItem rowSpan={3} colSpan={6}>
            <Plot
              data={traces}
              layout={{
                width: 1350,
                height: 740,
                hovermode: "closest",
                showlegend: true,
                margin: {
                  l: 0,
                  r: 0,
                  b: 0,
                  t: 0,
                },
                legend: {
                  title: {
                    text: "Categories",
                    font: {
                      size: 20,
                      color: backgroundColor,
                    },
                  },
                  font: {
                    size: 16,
                    color: backgroundColor,
                  },
                  orientation: "v",
                  itemsizing: "constant",
                  x: 0.9,
                  y: 0.5,
                },
                font: {
                  color: backgroundColor,
                },
                paper_bgcolor: gray,
              }}
            />
          </GridItem>
          <GridItem rowSpan={1} colSpan={1}></GridItem>

          <GridItem rowSpan={1} colSpan={1}>
            <VStack spacing={3} direction="row" align="center">
              <Text fontSize="2xl" color={textColor} align="center">
                Change page
              </Text>
              <Link to="/drawing" className="drawing_link">
                <Button
                  leftIcon={<BiPencil />}
                  color={backgroundColor}
                  border="2px"
                  borderColor={darkGray}
                  variant="solid"
                  size="lg"
                  height={buttonHeight}
                  width={buttonWidth}
                >
                  {" "}
                  Draw
                </Button>
              </Link>
              <Link to="/" className="explore_link">
                <Button
                  leftIcon={<BiImages />}
                  color={backgroundColor}
                  border="2px"
                  borderColor={darkGray}
                  variant="solid"
                  size="lg"
                  height={buttonHeight}
                  width={buttonWidth}
                >
                  {" "}
                  Dataset
                </Button>
              </Link>
            </VStack>
          </GridItem>

          <Text fontSize="xs" color={textColor} align="center"></Text>
        </Grid>
      </Box>
    </ChakraProvider>
  );
}

export default Embeddings;
