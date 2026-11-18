# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        #features.divideAll(10.0)
        return features

def closestTarget(pos, targets, walls):
    """
    BFS расстояние до ближайшей целевой позиции от pos.
    targets — список координат (x, y).
    """
    if not targets:
        return None
    queue = [(pos[0], pos[1], 0)]
    expanded = set()
    while queue:
        pos_x, pos_y, dist = queue.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if (pos_x, pos_y) in targets:
            return dist
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            queue.append((nbr_x, nbr_y, dist + 1))
    return None

# Вводим свои знания для обобщения состояний 
class MyExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghost_states = state.getGhostStates()
        capsules = state.getCapsules()
        features = util.Counter()

        # Найти направления движения
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Делим на 2 призраков
        active_ghosts = []
        scared_ghosts = []
        for ghost in ghost_states:
            if ghost.scaredTimer > 0:
                scared_ghosts.append(ghost)
            else:
                active_ghosts.append(ghost)

        active_ghost_positions = [g.getPosition() for g in active_ghosts]
        scared_ghost_positions = [(int(g.getPosition()[0]), int(g.getPosition()[1])) for g in scared_ghosts]

        features["bias"] = 1.0

        # Количество ОПАСНЫХ призраков в 1 шаге (испуганных игнорируем)
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls)
            for g in active_ghost_positions
        )

        # Определяем режим: охота или обычный
        hunting_mode = len(scared_ghosts) > 0

        # Еда — сниженный приоритет в режиме охоты
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            if hunting_mode:
                # Уменьшаем приоритет еды — охота на призраков важнее
                features["closest-food"] = 0.5 * float(dist) / (walls.width * walls.height)
            else:
                features["closest-food"] = float(dist) / (walls.width * walls.height)

        if hunting_mode:
            # РЕЖИМ ОХОТЫ: агрессивно гонемся за испуганными призраками в пределах 6 шагов
            # BFS расстояние до ближайшего испуганного призрака (учитывает стены)
            scared_dist = closestTarget((next_x, next_y), scared_ghost_positions, walls)
            if scared_dist is not None:
                if scared_dist <= 6:
                    # Призрак близко — сильный сигнал погони
                    features["closest-scared-ghost"] = float(scared_dist) / (walls.width * walls.height)
                else:
                    # Призрак далеко — слабый сигнал
                    features["closest-scared-ghost"] = 0.5 * float(scared_dist) / (walls.width * walls.height)
            # Проверяем — попадаем ли мы на испуганного призрака или рядом с ним
            for g in scared_ghost_positions:
                if (next_x, next_y) == g:
                    features["eats-ghost"] = 1.0
                    break
                if (next_x, next_y) in Actions.getLegalNeighbors(g, walls):
                    features["eats-ghost"] = 0.5
                    break

        else:
            # ОБЫЧНЫЙ РЕЖИМ: ищем капсулы и избегаем тупиков
            if capsules:
                capsule_dist = closestTarget((next_x, next_y), capsules, walls)
                if capsule_dist is not None:
                    features["closest-capsule"] = float(capsule_dist) / (walls.width * walls.height)

                # Съесть капсулу — очень ценно
                if (next_x, next_y) in capsules:
                    features["eats-capsule"] = 1.0

            # Избегаем тупиков — считаем выходы из следующей позиции
            legal_next = Actions.getLegalNeighbors((next_x, next_y), walls)
            if len(legal_next) <= 2:
                # Тупик или коридор — штрафуем если там нет еды
                if not food[next_x][next_y]:
                    features["dead-end"] = 1.0

        # Нормализуем все фичи чтобы веса не расходились
        features.divideAll(10.0)
        return features
