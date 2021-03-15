import { Box, ChakraProvider } from '@chakra-ui/react'
import { useSvgDrawing } from 'react-hooks-svgdrawing'

const App = () => {
  const [renderRef, draw] = useSvgDrawing({
    penWidth: 10
  })

  return <ChakraProvider>
    <Box
      h="100vh"
      w="100vw"
      ref={renderRef}
    />
  </ChakraProvider>
}

export default App
