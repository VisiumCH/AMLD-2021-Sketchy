import React, { useState, useCallback, useEffect } from 'react'
import { Link } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { Box, ChakraProvider, Button, Text, Heading, VStack } from '@chakra-ui/react'

const darkGray = "#A3A8B0"
const textColor = "#FFFFFF"
const backgroundColor = "#1A365D"
const buttonHeight = "48px"
const buttonWidth = "180px"

const colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']


function Embeddings() {
    const [isSending, setIsSending] = useState(false)
    const [result, setResult] = useState({})
    let traces = []

    async function getEmbeddings() {
        // Send to back end
        const response = await fetch('/get_embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ "body": "body" })
        })

        if (response.ok) {
            const res = await response.json()
            console.log(res)
            setResult(res)
            for (let key in res) {
                console.log(key)
                // console.log(res[key]['x'])
                const trace1 = {
                    x: [1, 2, 4],
                    y: [4, 5, 6],
                    z: [7, 8, 9],
                    name: 'trace1',
                    type: 'scatter3d',
                    mode: 'markers',
                    marker: {
                        color: 'red',
                        size: 4
                    }
                }
                const trace2 = {
                    x: [7, 2, 4],
                    y: [8, 5, 1],
                    z: [9, 6, 9],
                    name: 'trace2',
                    type: 'scatter3d',
                    mode: 'markers',
                    marker: {
                        color: 'green',
                        size: 4
                    }
                }
                traces = [trace1, trace2]
            }
        }
    }


    const sendRequest = useCallback(async (svg) => {
        // don't send again while we are sending
        if (isSending) return
        // update state
        setIsSending(true)

        // set images and labels
        getEmbeddings()

        // once the request is sent, update state again
        setIsSending(false)

    }, [isSending]) // update the callback if the state changes



    useEffect(() => {
        sendRequest()
    }, [sendRequest])

    let i = 0
    for (let key in result) {
        let trace = {
            x: result[key]['x'],
            y: result[key]['y'],
            z: result[key]['z'],
            name: key,
            type: 'scatter3d',
            mode: 'markers',
            marker: {
                color: colors[i],
                size: 4
            }
        }
        traces.push(trace)
        i = i + 1
    }

    return (
        <ChakraProvider >
            <Box bg={backgroundColor}>
                <VStack
                    spacing={4}
                    align="center"
                >
                    <Heading fontSize="4xl" color={textColor} align="center">
                        AMLD 2021 Visium's Sketchy App
                </Heading>
                    <Text fontSize="xs" color={textColor} align="center">
                        --------------------------------------------------------
                </Text>
                    <Text fontSize="xl" color={textColor} align="center">
                        Embeddings: Images and Sketches in latent space
                </Text>
                    <Plot
                        data={traces}
                        layout={{
                            width: 1200,
                            height: 650,
                            showlegend: true
                        }}
                    />
                    <Button color={backgroundColor} border="2px" borderColor={darkGray} variant="solid" size="lg" height={buttonHeight} width={buttonWidth} onClick={() => {
                        sendRequest()
                    }}>
                        Load Graph
                    </Button>
                    <Link to="/drawing" className="drawing_link">
                        <Button color={backgroundColor} border="2px" borderColor={darkGray} variant="solid" size="lg"> Back to Drawing</Button>
                    </Link>
                    <Text fontSize="xs" color={textColor} align="center">

                    </Text>
                </VStack>
            </Box>
        </ChakraProvider >
    )
}

export default Embeddings
