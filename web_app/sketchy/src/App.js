import React, { useEffect, useState, useCallback } from 'react'
import { Box, ChakraProvider, Button, Stack, Text, Heading, HStack, StackDivider, Grid, GridItem } from '@chakra-ui/react'
import { useSvgDrawing } from 'react-hooks-svgdrawing'



const App = () => {
  const [isSending, setIsSending] = useState(false)

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

  useEffect(() => {
    fetch('/api_list').then(response => console.log(response.json()))
  }, [])

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
      console.log('response worked in sendRequest')
      const res = await response.json()
      console.log(res)
      // const label = res["image_label"]
      // const image_base64 = res["image_base64"]

      // const Example = ({ image_base64 }) => <img src={`data:image/jpeg;base64,${image_base64}`} />
      // ReactDOM.render(<Example data={image_base64} />, document.getElementById('container'))

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
      <Grid
        h="100vh"
        w="100vw"
        templateRows="repeat(12, 1fr)"
        templateColumns="repeat(6, 1fr)"
        gap={4}
        align="center"
      >
        <GridItem rowSpan={1} colSpan={4}  >
          <Text fontSize="5xl" color="teal">
            Draw Sketch Here:
          </Text>
        </GridItem>
        <GridItem rowSpan={1} colSpan={2}  >
          <Text fontSize="5xl" color="teal">
            Infered Image:
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
            borderRadius="md"
            ref={divRef}
          />
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
          <Text fontSize="3xl" color="teal">
            Good guess ?
          </Text>
        </GridItem>

      </Grid>
    </Stack>

  </ChakraProvider >
}

export default App
