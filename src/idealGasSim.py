import numpy as np
import math
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


def particle_collision_time(pos_i,
                            vel_i,
                            pos_j,
                            vel_j,
                            particle_radius,
                            large_num):
    delta_x = pos_j[0] - pos_i[0]
    delta_vx = vel_j[0] - vel_i[0]
    delta_y = pos_j[1] - pos_i[1]
    delta_vy = vel_j[1] - vel_i[1]

    a = delta_vx**2 + delta_vy**2
    b = 2 * (delta_x * delta_vx + delta_y * delta_vy)
    c = delta_x**2 + delta_y**2 - 4 * particle_radius**2  # Adjusting for radius    
    if a == 0:
        return large_num  # Particles moving parallel (no collision)

    Delta = b**2 - 4 * a * c
    if Delta < 0:
        return large_num  # No real roots, no collision
    elif Delta == 0:
        t = -b / (2 * a)
        return t if t >= 0 else large_num
    else:
        t1 = (-b + math.sqrt(Delta)) / (2 * a)
        t2 = (-b - math.sqrt(Delta)) / (2 * a)
        if t1 >= 0 and t2 >= 0:
            return min(t1, t2)
        elif t1 >= 0:
            return t1
        elif t2 >= 0:
            return t2
        return large_num


# Function to compute the time to collision with the walls
def wall_collision_time(position,
                        velocity,
                        box_size,
                        particle_radius,
                        large_num):
    times = []
    for i in range(2):  # x and y directions
        if velocity[i] > 0:
            t_wall = (box_size - particle_radius - position[i]) / velocity[i]
        elif velocity[i] < 0:
            t_wall = (particle_radius - position[i]) / velocity[i]
        else:
            t_wall = large_num  # No movement in this direction
        times.append(t_wall)

    return min(times)


# Function to find the next collision event
def find_earliest_collision(positions,
                            velocities,
                            num_particles,
                            large_num,
                            box_size,
                            particle_radius):
    min_time = large_num
    collision_pair = (-1, -1)  # (-1, -1) means wall collision

    # Check particle-particle collisions
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            t_collision = particle_collision_time(positions[i], velocities[i], positions[j], velocities[j], particle_radius, large_num)
            if t_collision < min_time:
                min_time = t_collision
                collision_pair = (i, j)

    # Check particle-wall collisions
    for i in range(num_particles):
        t_wall = wall_collision_time(positions[i], velocities[i], box_size, particle_radius, large_num)
        if t_wall < min_time:
            min_time = t_wall
            collision_pair = (i, -1)
            # Indicate wall collision for this particle

    return min_time, collision_pair


def update_positions(positions,
                     velocities,
                     dt,
                     collision_pair,
                     pressure, box_size,
                     num_particles,
                     particle_radius):
    # Update positions
    positions += velocities * dt
    for i in range(num_particles):
        for j in range(2):  # x and y directions
            if positions[i, j] <= particle_radius or positions[i, j] >= box_size - particle_radius:
                velocities[i, j] *= -1
                
                pressure += abs(velocities[i, j])
                positions[i, j] = np.clip(positions[i, j], particle_radius, box_size - particle_radius)

    # Check for particle collisions
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= 2 * particle_radius:  # Collision condition
                velocities = resolve_collision(i, j, positions, velocities)
    positions += velocities * 0.000001
    
    return pressure, positions, velocities


# Function to resolve the collision (particles or walls)
def resolve_collision(i, j, positions, velocities):
    # Calculate the normal vector
    delta_pos = positions[i] - positions[j]
    dist = np.linalg.norm(delta_pos)
    delta_v = velocities[i] - velocities[j]
    # Normal vector
    n_hat = delta_pos / (dist**2)
    # Relative velocity

    # Velocity component along the normal direction
    v_rel = np.dot(delta_v, delta_pos)

    # Update velocities if particles are moving towards each other
    if v_rel < 0:
        velocities[i] -= v_rel * n_hat
        velocities[j] += v_rel * n_hat
    return velocities


# Simulation function, returns pressure
def molecular_sim(box_dim, num_particles, particle_radius, t_max, large_num, positions, velocities):
    box_size = box_dim
    pressure = 0
    # Initialize positions and velocities

    # Function to compute the time to the first collision between two particles

    t = 0
    while t < t_max:
        delta_t, collision_pair = find_earliest_collision(positions, velocities,num_particles, large_num, box_size, particle_radius)

        delta_t += 0.000001
        pressure, positions, velocities = update_positions(positions, velocities, delta_t, collision_pair, pressure, box_size, num_particles, particle_radius)
        t += (delta_t + 0.000001)
    return pressure
