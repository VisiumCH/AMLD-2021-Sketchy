import React, { useEffect, useState, useCallback } from 'react'
import ReactDOM from "react-dom"
import { Box, ChakraProvider, Button, Stack, Text, Heading, Image, StackDivider, Grid, GridItem } from '@chakra-ui/react'
import { useSvgDrawing } from 'react-hooks-svgdrawing'



const App = () => {
  const [isSending, setIsSending] = useState(false)
  const [inferredImage, setInferredImage] = useState([])
  const [inferredLabel, setInferredLabel] = useState([])

  const [
    divRef,
    {
      getSvgXML,
      download,
      undo,
      clear
    }
  ] = useSvgDrawing({
    penWidth: 8,     // pen width
    penColor: '#000000', // pen color
    width: 300, // drawing area width
    height: 300 // drawing area height
  })

  const sendRequest = useCallback(async (svg) => {
    // don't send again while we are sending
    if (isSending) return
    // update state
    setIsSending(true)

    // send the actual request
    const response = await fetch('/find_images', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ "sketch": svg })
    })
    if (response.ok) {
      const res = await response.json()
      console.log(res)
      let inferredImages = res["images_base64"]
      let tempImage = ''
      for (let i = 0; i < inferredImages.length; i++) {
        tempImage = inferredImages[i].split('\'')[1]
        inferredImages[i] = <img src={`data:image/jpeg;base64,${tempImage}`} />
      }
      setInferredImage(inferredImages)
      setInferredLabel(res["images_label"])
    } else {
      console.log('Response did not worked in sendRequest')
    }
    // once the request is sent, update state again
    setIsSending(false)
  }, [isSending]) // update the callback if the state changes

  return <ChakraProvider >
    <Stack
      spacing="0px"
      align="center">
      <Heading fontSize="4xl" color="teal">
        Sketchy App
      </Heading>
      <Grid
        h="95vh"
        w="99vw"
        templateRows="repeat(12, 1fr)"
        templateColumns="repeat(6, 1fr)"
        gap={4}
        align="center"
      >
        <GridItem rowSpan={1} colSpan={4}  >
          <Text fontSize="4xl" color="teal">
            Draw Sketch Here:
          </Text>
        </GridItem>
        <GridItem rowSpan={1} colSpan={2}  >
          <Text fontSize="4xl" color="teal">
            Inferred Images:
          </Text>
        </GridItem>

        <GridItem rowSpan={9} colSpan={4} >
          <Box
            h="70vh"
            w="60vw"
            bg="#d0d5d9"
            borderRadius="md"
            ref={divRef}
          />
        </GridItem>
        <GridItem rowSpan={9} colSpan={2}  >
          <Box
            h="70vh"
            w="30vw"
            bg="#d0d5d9"
            borderRadius="md">
            <Text fontSize="2xl" color="teal">
              1. Class guessed: {inferredLabel[0]}
            </Text>
            {inferredImage[0]}

            <Text fontSize="2xl" color="teal">
              2. Class guessed: {inferredLabel[1]}
            </Text>
            {inferredImage[1]}
          </Box>
        </GridItem>
        <GridItem rowSpan={2} >
          <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() =>
            undo()
          }>
            Undo last line
        </Button>
        </GridItem>
        <GridItem rowSpan={2} >
          <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() =>
            clear()
          }>
            Erase everything
        </Button>
        </GridItem>
        <GridItem rowSpan={2}  >
          <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() =>
            sendRequest(getSvgXML())
          }>
            Send Sketch
        </Button>
        </GridItem>
        <GridItem rowSpan={2} >
          <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() =>
            download('png')
          }>
            Download Sketch
        </Button>
        </GridItem>
        <GridItem rowSpan={1} colSpan={2} color='blue' >
          <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() => {
            clear()
            setInferredImage([])
            setInferredLabel([])
          }
          }>
            Play Again!
          </Button>
        </GridItem>

      </Grid>
    </Stack>

  </ChakraProvider >
}

export default App
