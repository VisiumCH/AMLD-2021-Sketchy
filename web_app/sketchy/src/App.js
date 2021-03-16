import React, { useEffect, useState, useRef, useCallback } from 'react'
import { Box, ChakraProvider, Button, Stack, Heading, HStack, StackDivider } from '@chakra-ui/react'
import { useSvgDrawing } from 'react-hooks-svgdrawing'



const App = () => {
  const [isSending, setIsSending] = useState(false)
  const isMounted = useRef(true)

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

  // set isMounted to false when we unmount the component
  useEffect(() => {
    return () => {
      isMounted.current = false
    }
  }, [])

  const sendSketch = useCallback(async (svg) => {
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
      console.log('response worked')
      const res = response.json()
      console.log(res)
      console.log(res["image_label"])

    } else {
      console.log('response did not worked')
    }

    // once the request is sent, update state again
    if (isMounted.current) // only update if we are still mounted
      setIsSending(false)
  }, [isSending]) // update the callback if the state changes




  return <ChakraProvider >
    <Stack
      spacing={4}
      align="center">
      <Heading fontSize="5xl" color="teal">
        Draw Here:
      </Heading>
      <Box
        h="75vh"
        w="90vw"
        bg="#d0d5d9"
        borderRadius="md"
        ref={divRef}
      />

      <HStack
        divider={<StackDivider borderColor="gray.200" />}
        spacing={4}
        align="stretch"
      >
        <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() =>
          undo()
        }>
          Undo last line
        </Button>
        <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() =>
          clear()
        }>
          Erase everything
        </Button>
      </HStack>

      <HStack
        divider={<StackDivider borderColor="gray.200" />}
        spacing={4}
        align="stretch"
      >
        <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() =>
          sendSketch(getSvgXML())
        }>
          Send Sketch
        </Button>
        <Button colorScheme="teal" size="lg" height="48px" width="160px" onClick={() =>
          download('png')
        }>
          Download Sketch
        </Button>
      </HStack>
    </Stack>
  </ChakraProvider >
}

export default App
