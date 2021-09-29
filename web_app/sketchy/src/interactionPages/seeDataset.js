import React, { useState, useCallback, useEffect } from "react";
import {
  Box,
  Text,
  Select,
  IconButton,
  Grid,
  GridItem,
  FormControl,
  HStack,
} from "@chakra-ui/react";
import { datasets, categories, nb_to_show } from "../constants";
import { heading } from "../ui_utils";
import { PageDrawer } from "../drawer.js";
import { BiRefresh } from "react-icons/bi";

function Dataset() {
  const [isSending, setIsSending] = useState(false);
  const [currentDataset, setCurrentDataset] = useState("Quickdraw");
  const [currentCategory, setCurrentCategory] = useState("pineapple");
  const [images, setImages] = useState([]);
  const [sketches, setSketches] = useState([]);
  const categoriesOptions = categories[currentDataset].map((category) => (
    <option>{category}</option>
  ));
  const datasetsOptions = datasets.map((dataset) => <option>{dataset}</option>);

  async function getImages(currentCategory, currentDataset) {
    // Send to back end
    const response = await fetch("/api/get_dataset_images", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        category: currentCategory,
        dataset: currentDataset,
      }),
    });

    if (response.ok) {
      let tmpImages = [];
      let tmpSketches = [];
      let tmpImage = "";
      let tmpSketch = "";
      const res = await response.json();
      for (let i = 0; i < nb_to_show; i++) {
        tmpImage = res["images_" + i + "_base64"].split("'")[1];
        tmpImages[i] = (
          <img
            src={`data:image/jpeg;base64,${tmpImage}`}
            alt={"image_" + i}
            style={{ height: "250px" }}
          />
        );
        tmpSketch = res["sketches_" + i + "_base64"].split("'")[1];
        tmpSketches[i] = (
          <img
            src={`data:image/jpeg;base64,${tmpSketch}`}
            alt={"sketch_" + i}
            style={{ height: "250px" }}
          />
        );
      }
      setImages(tmpImages);
      setSketches(tmpSketches);
    }
  }

  const sendRequest = useCallback(
    async (currentCategory, currentDataset) => {
      if (isSending) return;
      setIsSending(true);
      getImages(currentCategory, currentDataset);
      setIsSending(false);
    },
    [isSending]
  );

  useEffect(() => {
    sendRequest(currentCategory, currentDataset);
  }, [sendRequest, currentCategory, currentDataset]);

  const renderImages = images.map((image) => (
    <Box w="50vw" h="30vh">
      {image}
    </Box>
  ));

  const renderSketches = sketches.map((sketch) => (
    <Box w="50vw" h="30vh">
      {sketch}
    </Box>
  ));

  return (
    <>
      {heading}
      <Text fontSize="xs">.</Text>
      <Grid
        h="91vh"
        gap={4}
        align="center"
        templateRows="repeat(11, 1fr)"
        templateColumns="repeat(2, 1fr)"
      >
        <GridItem rowSpan={1} colSpan={1} align="right">
          <Text fontSize="2xl" color="white">
            Choose a dataset:
          </Text>
        </GridItem>
        <GridItem rowSpan={1} colSpan={1} align="left">
          <form>
            <FormControl
              id="category"
              color="darkBlue"
              width="200px"
              bg="white"
            >
              <Select
                value={currentDataset}
                color="darkBlue"
                onChange={(e) => {
                  setCurrentDataset(e.target.value);
                  setCurrentCategory("pineapple");
                }}
              >
                {datasetsOptions}
              </Select>
            </FormControl>
          </form>
        </GridItem>
        <GridItem rowSpan={1} colSpan={1} align="right">
          <Text fontSize="2xl" color="white">
            Choose a categories to see some samples:
          </Text>
        </GridItem>
        <GridItem rowSpan={1} colSpan={1} align="left">
          <HStack>
            <form>
              <FormControl
                id="category"
                color="darkBlue"
                width="200px"
                bg="white"
              >
                <Select
                  value={currentCategory}
                  color="darkBlue"
                  onChange={(e) => setCurrentCategory(e.target.value)}
                >
                  {categoriesOptions}
                </Select>
              </FormControl>
            </form>
            <IconButton
              color="white"
              icon={<BiRefresh />}
              border="2px"
              borderColor="darkGray"
              variant="solid"
              size="md"
              onClick={() => {
                getImages(currentCategory, currentDataset);
              }}
            ></IconButton>
          </HStack>
        </GridItem>
        <GridItem rowSpan={4} colSpan={2}>
          <HStack>{renderImages}</HStack>
        </GridItem>
        <GridItem rowSpan={4} colSpan={2}>
          <HStack>{renderSketches}</HStack>
        </GridItem>
        <GridItem rowSpan={1} colSpan={2}>
          {PageDrawer("undefined")}
        </GridItem>
      </Grid>
    </>
  );
}

export default Dataset;
