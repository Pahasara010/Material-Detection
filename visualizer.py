#!/usr/bin/env python3

import asyncio  # For handling asynchronous WebSocket communication and animation loop
import pygame  # For creating the graphical visualization with a box
import json  # For parsing WebSocket messages containing material data
import websockets  # For connecting to the WebSocket server to receive material predictions
import platform  # For checking if running on Emscripten (Pyodide in browser)

# Constants for the visualization window
WIDTH, HEIGHT = 500, 500  # Set window dimensions to 500x500 pixels
WS_SERVER = "ws://127.0.0.1:9999"  # WebSocket server address to receive material data
FPS = 30  # Frames per second for smooth animation

# Initialize pygame for visualization
pygame.init()  # Initialize Pygame library
screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Create a window of specified size
pygame.display.set_caption("📦 CSI Material Visualizer")  # Set window title with emoji
clock = pygame.time.Clock()  # Create a clock to control frame rate
font = pygame.font.SysFont("Arial", 24)  # Initialize font for rendering action text

# Global variables to store the current detected container and material
current_container = "unknown"
current_material = "unknown"  # Default when no data is received

# Function to draw a box representing the container and material
def draw_container(center_x, center_y, container="unknown", material="unknown"):
    screen.fill((240, 240, 240))  # Fill background with a light gray color

    # Define colors for materials
    material_colors = {
        'iron': (100, 100, 100),  # Gray
        'wood': (139, 69, 19),    # Brown
        'plastic': (200, 200, 200),  # Light gray
        'water': (0, 0, 255),     # Blue
        'fabric': (255, 0, 0),    # Red
    }

    # Define colors for container borders
    container_colors = {
        'cardboard': (139, 69, 19),  # Brown
        'plastic': (211, 211, 211),  # Light gray
    }

    # Draw the container as a rectangle border
    box_rect = pygame.Rect(center_x - 100, center_y - 100, 200, 200)
    box_color = container_colors.get(container, (0, 0, 0))
    pygame.draw.rect(screen, box_color, box_rect, 4)  # Thicker border for visibility

    # Fill inner rectangle with material color
    inner_rect = box_rect.inflate(-40, -40)
    material_color = material_colors.get(material, (255, 255, 255))
    pygame.draw.rect(screen, material_color, inner_rect)

    # Display the current container and material as text on the screen
    label_container = font.render(f"Container: {container.title()}", True, (20, 20, 20))
    label_material = font.render(f"Material: {material.title()}", True, (20, 20, 20))
    screen.blit(label_container, (20, 20))
    screen.blit(label_material, (20, 50))

# WebSocket client to receive material predictions
async def listen_ws():
    # Continuously listen for WebSocket messages and update the current container and material
    global current_container, current_material
    while True:
        try:
            async with websockets.connect(WS_SERVER) as websocket:  # Connect to the WebSocket server
                print("✅ Connected to WebSocket")
                while True:
                    message = await websocket.recv()  # Receive message from server
                    data = json.loads(message)  # Parse JSON message
                    hyp = data["hypothesis"]  # Get the predicted hypothesis (container_material)
                    if hyp and "_" in hyp:
                        parts = hyp.split("_")
                        current_container = parts[0]
                        current_material = parts[1]
                    else:
                        current_container = "unknown"
                        current_material = "unknown"
        except Exception as e:
            print(f"[!] WebSocket error: {e}")  # Log any WebSocket connection errors
            current_container = "unknown"
            current_material = "unknown"  # Fallback on error
            await asyncio.sleep(2)  # Wait 2 seconds before retrying

# Asynchronous loop to run the Pygame visualization
async def run_visualizer():
    # Draw the container and material
    while True:
        for event in pygame.event.get():  # Handle Pygame events
            if event.type == pygame.QUIT:  # Check for window close event
                pygame.quit()  # Quit Pygame
                return

        draw_container(WIDTH // 2, HEIGHT // 2, container=current_container, material=current_material)  # Draw at center
        pygame.display.flip()  # Update the display
        clock.tick(FPS)  # Control frame rate
        await asyncio.sleep(0.01)  # Yield control for a short time

# Main coroutine to run WebSocket listener and visualizer concurrently
async def main():
    # Run the WebSocket listener and visualizer tasks together
    await asyncio.gather(
        listen_ws(),
        run_visualizer()
    )

# Entry point for the script
if platform.system() == "Emscripten":  # Check if running in Pyodide (browser)
    asyncio.ensure_future(main())  # Schedule the main coroutine for Pyodide
else:
    if __name__ == "__main__":
        try:
            asyncio.run(main())  # Run the main coroutine for desktop Python
        except KeyboardInterrupt:
            print("👋 Exiting visualizer")  # Log exit on user interrupt
            pygame.quit()  # Clean up Pygame resources