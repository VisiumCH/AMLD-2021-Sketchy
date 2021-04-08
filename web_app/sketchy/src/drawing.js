import React, { useState, useCallback, useEffect } from "react";
// import { Link as ReachLink } from 'react-router-dom'
import { Link } from "react-router-dom";
import {
  Box,
  ChakraProvider,
  Button,
  HStack,
  VStack,
  Text,
  Heading,
  Grid,
  GridItem,
  CircularProgress,
} from "@chakra-ui/react";
import { useSvgDrawing } from "react-hooks-svgdrawing";
import {
  gray,
  darkGray,
  textColor,
  backgroundColor,
  buttonHeight,
  buttonWidth,
  white,
  black,
} from "./constants";
import { BiPencil, BiEraser } from "react-icons/bi";

const progress = (
  <CircularProgress
    isIndeterminate
    color={backgroundColor}
    size="180px"
    thickness="4px"
  />
);

function Drawing() {
  const [isSending, setIsSending] = useState(false);
  const [inferredImage, setInferredImage] = useState([]);
  const [inferredLabel, setInferredLabel] = useState([]);
  const [attention, setAttention] = useState("");
  const [svg, setSvg] = useState("");
  const [drawingMode, setDrawingMode] = useState("drawing")
  const [disableDrawing, setDisableDrawing] = useState(true)
  const [disableErasing, setDisableErasing] = useState(false)


  const [divRef, { getSvgXML, changePenColor, changePenWidth, undo, clear }] = useSvgDrawing({
    penWidth: 3, // pen width (similar as database width)
    penColor: black, // pen color
    width: 300, // drawing area width
    height: 300, // drawing area height
  });

  function changeDrawingMode() {
    if (drawingMode === "drawing") {
      changePenColor(white)
      changePenWidth(25)
      setDrawingMode("erasing")
      setDisableDrawing(false)
      setDisableErasing(true)
    }
    if (drawingMode === "erasing") {
      changePenColor(black)
      changePenWidth(3)
      setDrawingMode("drawing")
      setDisableDrawing(true)
      setDisableErasing(false)
    }
  }

  async function setInference(svg) {
    // Check that there is visible data in the svg
    console.log(svg)
    if (svg == "null" || svg.length < 500) {
      setInferredImage([]);
      setInferredLabel([]);
      setAttention("");
      return;
    }

    // Show that we are processing the request
    setInferredImage([progress, progress]);
    setInferredLabel(["Guess 1: ???", "Guess 2: ???"]);
    setAttention(progress);

    // Send to back end
    const response = await fetch("/find_images", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ sketch: svg }),
    });

    // Receive response
    if (response.ok) {
      const res = await response.json();
      let inferredImages = res["images_base64"];
      let inferredLabels = res["images_label"];
      let tempImage = "";
      for (let i = 0; i < inferredImages.length; i++) {
        tempImage = inferredImages[i].split("'")[1];
        inferredImages[i] = (
          <img
            src={`data:image/jpeg;base64,${tempImage}`}
            alt="inferred_image"
          />
        );
        inferredLabels[i] = `Guess ${i + 1}: ${inferredLabels[i]}`;
      }
      setInferredImage(inferredImages);
      setInferredLabel(inferredLabels);

      let tempAttention = res["attention"].split("'")[1];
      setAttention(
        <img
          src={`data:image/jpeg;base64,${tempAttention}`}
          alt="attention_image"
        />
      );
    }
  }

  const sendRequest = useCallback(
    async (svg) => {
      // don't send again while we are sending
      if (isSending) return;
      // update state
      setIsSending(true);

      // set images and labels
      setInference(svg);

      // once the request is sent, update state again
      setIsSending(false);
    },
    [isSending]
  ); // update the callback if the state changes

  useEffect(() => {
    sendRequest(svg);
  }, [sendRequest, svg]);

  return (
    <ChakraProvider>
      <Box bg={backgroundColor}>
        <Heading fontSize="4xl" color={textColor} align="center">
          AMLD 2021 Visium's Sketchy App
        </Heading>
        <Text fontSize="xs" color={textColor} align="center">
          --------------------------------------------------------
        </Text>

        <Grid
          h="96vh"
          w="98vw"
          gap={4}
          align="center"
          templateRows="repeat(14, 1fr)"
          templateColumns="repeat(12, 1fr)"
        >
          <GridItem rowSpan={1} colSpan={8}>
            <Text fontSize="4xl" color={textColor}>
              {" "}
              Draw Sketch Here:
            </Text>
          </GridItem>

          <GridItem rowSpan={1} colSpan={4}>
            <Text fontSize="4xl" color={textColor}>
              {" "}
              Closest Images:{" "}
            </Text>
          </GridItem>

          <GridItem rowSpan={11} colSpan={8}>
            <Box
              h="74vh"
              w="62vw"
              bg={white}
              borderWidth="5px"
              borderRadius="lg"
              borderColor="#A3A8B0"
              ref={divRef}
              _hover={{
                //TODO: change cursor when drawing or erasing
                //cursor: "url(https://i.stack.imgur.com/bUGV0.png),auto"
              }}
              // onTouchEnd={() => sendRequest(getSvgXML())} // touch screen
              onMouseMove={() => {
                setSvg(getSvgXML());
              }}
            ></Box>
          </GridItem>

          <GridItem rowSpan={5} colSpan={4}>
            <Box>
              <HStack
                align="center"
                borderWidth="5px"
                borderRadius="lg"
                borderColor="#A3A8B0"
                bg={gray}
              >
                <Box h="33vh" w="20vw">
                  <VStack>
                    <Text fontSize="2xl" color={backgroundColor} as="em">
                      {inferredLabel[0]}
                    </Text>
                    <Box w="100%" h="35%">
                      {inferredImage[0]}
                    </Box>
                  </VStack>
                </Box>
                <Box h="33vh" w="20vw">
                  <VStack>
                    <Text fontSize="2xl" color={backgroundColor} as="em">
                      {inferredLabel[1]}
                    </Text>
                    <Box bg={gray} w="100%" h="35%">
                      {inferredImage[1]}
                    </Box>
                  </VStack>
                </Box>
              </HStack>
            </Box>
          </GridItem>

          <GridItem rowSpan={1} colSpan={4} bg={backgroundColor}>
            <Text fontSize="4xl" color={textColor}>
              Attention Map
            </Text>
          </GridItem>

          <GridItem
            rowSpan={5}
            colSpan={4}
            borderWidth="5px"
            borderRadius="lg"
            borderColor="#A3A8B0"
            bg={gray}
          >
            <Box h="35vh" w="21vw">
              {attention}
            </Box>
          </GridItem>

          <GridItem rowSpan={2} colSpan={2}>
            <Button
              disabled={disableDrawing}
              leftIcon={<BiPencil />}
              color={backgroundColor}
              border="2px"
              borderColor={darkGray}
              variant="solid"
              size="lg"
              height={buttonHeight}
              width={buttonWidth}
              onClick={() => {
                if (drawingMode === "erasing") {
                  changeDrawingMode()
                }
              }}
            >
              Drawing
            </Button>
          </GridItem>
          <GridItem rowSpan={2} colSpan={2}>
            <Button
              disabled={disableErasing}
              leftIcon={<BiEraser />}
              color={backgroundColor}
              border="2px"
              borderColor={darkGray}
              variant="solid"
              size="lg"
              height={buttonHeight}
              width={buttonWidth}
              onClick={() => {
                if (drawingMode === "drawing") {
                  changeDrawingMode()
                }
              }}
            >
              Erasing
            </Button>
          </GridItem>
          <GridItem rowSpan={2} colSpan={2}>
            <Button
              color={backgroundColor}
              border="2px"
              borderColor={darkGray}
              variant="solid"
              size="lg"
              height={buttonHeight}
              width={buttonWidth}
              onClick={() => {
                undo();
                sendRequest(getSvgXML());
              }}
            >
              Undo
            </Button>
          </GridItem>
          <GridItem rowSpan={2} colSpan={2}>
            <Button
              color={backgroundColor}
              border="2px"
              borderColor={darkGray}
              variant="solid"
              size="lg"
              height={buttonHeight}
              width={buttonWidth}
              onClick={() => {
                clear();
                setInferredImage([]);
                setInferredLabel([]);
                setAttention("");
              }}
            >
              Restart!
            </Button>
          </GridItem>
          <GridItem rowSpan={2} colSpan={4}>
            <Link
              to={{
                pathname: "/embeddings",
                state: svg,
              }}
            >
              <Button
                color={backgroundColor}
                border="2px"
                borderColor={darkGray}
                variant="solid"
                size="lg"
              >
                {" "}
                See Embeddings
              </Button>
            </Link>
          </GridItem>
        </Grid>
      </Box>
    </ChakraProvider>
  );
}

export default Drawing;
